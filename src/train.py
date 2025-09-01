#!/usr/bin/env python3
"""
Model training module for Alzheimer's prediction.

- Loads featurized data
- Patient-level splits
- Feature selection
- Trains Logistic Regression and XGBoost
- Saves artifacts required by serving:
    * artifacts/latest/model.pkl
    * artifacts/latest/feature_names.json
    * artifacts/latest/threshold.json  ({"optimal_threshold": <float>})
    * artifacts/latest/metrics.json
- Creates dirs and applies 0o777 permissions where needed
"""

import logging
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pickle
import stat

import typer
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Experiment tracking utils (kept as in your code)
from utils import (
    log_metrics, log_artifact, log_table, log_plot,
    start_run, end_run, get_run_id, tracker, tracker_type
)

# New tracking context manager
from tracking import tracker_run


# ---------------------------
# Helpers: dirs & permissions
# ---------------------------

def ensure_dir(path: Path, mode: int = 0o777) -> Path:
    """Create directory (parents ok) and set permissive permissions."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except Exception:
        pass
    return path

def ensure_parent(file_path: Path, mode: int = 0o777) -> None:
    """Ensure parent directory exists with desired permissions."""
    ensure_dir(file_path.parent, mode)

def write_json(path: Path, data: Any, mode_file: int = 0o666, mode_dir: int = 0o777) -> None:
    """Write JSON atomically and apply permissions."""
    ensure_parent(path, mode_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    try:
        os.chmod(path, mode_file)
    except Exception:
        pass

def write_pickle(path: Path, obj: Any, mode_file: int = 0o666, mode_dir: int = 0o777) -> None:
    """Write pickle and apply permissions."""
    ensure_parent(path, mode_dir)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    try:
        os.chmod(path, mode_file)
    except Exception:
        pass


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data configuration
    input_dir: str = "/Data/featurized"  # Will be overridden by _load_data to check multiple locations
    target_column: str = "alzheimers_diagnosis"
    exclude_columns: List[str] = field(default_factory=lambda: ["patient_id", "year"])

    # Split configuration
    test_size: float = 0.2
    val_size: float = 0.2  # from training set
    random_state: int = 42
    stratify: bool = True

    # Feature selection
    max_features: int = 150
    variance_threshold: float = 0.01

    # Class imbalance
    handle_imbalance: str = "class_weight"  # "class_weight", "smote", "none"

    # Model configuration
    models: List[str] = field(default_factory=lambda: ["logistic_regression", "xgboost"])

    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_leaves": 32,
        "verbosity": 0
    })

    # Logistic Regression parameters
    lr_params: Dict[str, Any] = field(default_factory=lambda: {
        "random_state": 42,
        "max_iter": 1000
    })

    # Output configuration
    output_dir: str = "models"
    save_metadata: bool = True
    log_artifacts: bool = True


class ModelTrainer:
    """Handles model training with patient-level splits and feature selection."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.feature_names = []
        self.preprocessing_metadata = {}
        self.lr_scaler = None  # avoid attribute error

    def _generate_model_name(self, run_type: str = "initial", include_timestamp: bool = True) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")
        run_descriptions = {
            "initial": "Alzheimer's Detection - Initial Training",
            "balanced": "Alzheimer's Detection - Balanced Data",
            "feature_optimized": "Alzheimer's Detection - Feature Optimized",
            "hyperparameter_tuned": "Alzheimer's Detection - Hyperparameter Tuned",
            "final": "Alzheimer's Detection - Final Model",
            "production": "Alzheimer's Detection - Production Ready"
        }
        description = run_descriptions.get(run_type, f"Alzheimer's Detection - {run_type}")
        models_str = "+".join(self.config.models)
        features_str = f"{self.config.max_features}feat"
        if include_timestamp:
            return f"{description} | {models_str} | {features_str} | {current_date} {current_time}"
        else:
            return f"{description} | {models_str} | {features_str} | seed_{self.config.random_state}"

    def _load_data(self) -> pd.DataFrame:
        """Load featurized data from partitioned Parquet files."""
        possible_paths = [
            Path(self.config.input_dir),
            Path("/Data/featurized"),
        ]
        input_path = None
        for path in possible_paths:
            if path.exists():
                input_path = path
                print(f"📁 Using data from: {input_path}")
                break
        if input_path is None:
            raise FileNotFoundError(f"Input directory not found in any of: {[str(p) for p in possible_paths]}")

        lazy_frames = []
        year_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.isdigit()]
        for d in tqdm(year_dirs, desc="Loading data", unit="year", leave=False):
            year = int(d.name)
            lf = pl.scan_parquet(str(d / "*.parquet")).with_columns(pl.lit(year).alias("year"))
            lazy_frames.append(lf)

        if not lazy_frames:
            df = pl.scan_parquet(str(input_path / "**" / "*.parquet")).collect()
        else:
            df = pl.concat(lazy_frames, how="vertical").collect()

        df_pandas = df.to_pandas()
        df_pandas = self._handle_diagnosis_uncertainty(df_pandas)
        return df_pandas

    def _patient_level_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        patients = df['patient_id'].unique()
        if self.config.stratify:
            patient_targets = df.groupby('patient_id')[self.config.target_column].agg(
                lambda x: 1 if x.sum() > len(x) / 2 else 0
            )
            target_counts = patient_targets.value_counts()
            stratify = len(target_counts) >= 2 and target_counts.min() >= 2
        else:
            stratify = False

        if stratify:
            train_patients, test_patients = train_test_split(
                patients,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=patient_targets.loc[patients]
            )
            train_patients, val_patients = train_test_split(
                train_patients,
                test_size=self.config.val_size,
                random_state=self.config.random_state,
                stratify=patient_targets.loc[train_patients]
            )
        else:
            train_patients, test_patients = train_test_split(
                patients,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            train_patients, val_patients = train_test_split(
                train_patients,
                test_size=self.config.val_size,
                random_state=self.config.random_state
            )

        train_df = df[df['patient_id'].isin(train_patients)]
        val_df = df[df['patient_id'].isin(val_patients)]
        test_df = df[df['patient_id'].isin(test_patients)]
        return train_df, val_df, test_df

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        feature_cols = [c for c in df.columns if c not in self.config.exclude_columns + [self.config.target_column]]
        numeric_cols, categorical_cols = [], []
        for col in tqdm(feature_cols, desc="Analyzing features", unit="feat", leave=False):
            if df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        print(f"📊 Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

        if categorical_cols:
            max_categorical_cols = 50
            if len(categorical_cols) > max_categorical_cols:
                counts = {c: df[c].nunique() for c in categorical_cols}
                categorical_cols = [c for c, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_categorical_cols]]
            categorical_df = pd.get_dummies(df[categorical_cols])
            numeric_df = df[numeric_cols] if numeric_cols else pd.DataFrame()
            X = pd.concat([numeric_df, categorical_df], axis=1) if not numeric_df.empty else categorical_df
            feature_cols = list(X.columns)
        else:
            X = df[feature_cols]

        X_array = X.values.astype(float)
        y_array = df[self.config.target_column].values
        return X_array, y_array, feature_cols

    def _feature_selection(self, X_train: np.ndarray, feature_names: List[str], y_train: np.ndarray) -> List[str]:
        variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_train_var = variance_selector.fit_transform(X_train)
        var_features = [feature_names[i] for i in variance_selector.get_support(indices=True)]

        if len(var_features) > self.config.max_features:
            xgb_selector = xgb.XGBClassifier(
                n_estimators=25, max_depth=3, learning_rate=0.3,
                tree_method="hist", grow_policy="lossguide", max_leaves=16,
                verbosity=0, random_state=self.config.random_state, eval_metric='logloss'
            )
            var_indices = [feature_names.index(f) for f in var_features]
            X_train_var_subset = X_train[:, var_indices]
            xgb_selector.fit(X_train_var_subset, y_train)
            importance_scores = xgb_selector.feature_importances_
            feature_importance = list(zip(var_features, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            selected = [f for f, _ in feature_importance[:self.config.max_features]]

            self.preprocessing_metadata['feature_selection'] = {
                'variance_threshold': self.config.variance_threshold,
                'max_features': self.config.max_features,
                'variance_selected': len(var_features),
                'final_selected': len(selected),
                'feature_importance': dict(feature_importance[:self.config.max_features])
            }
            return selected
        else:
            self.preprocessing_metadata['feature_selection'] = {
                'variance_threshold': self.config.variance_threshold,
                'max_features': len(var_features),
                'variance_selected': len(var_features),
                'final_selected': len(var_features)
            }
            return var_features

    def _handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.handle_imbalance == "smote":
            smote = SMOTE(random_state=self.config.random_state)
            return smote.fit_resample(X, y)
        elif self.config.handle_imbalance == "class_weight":
            return X, y
        else:
            return X, y

    def _handle_diagnosis_uncertainty(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['years_since_first_symptoms'] = np.random.normal(3.5, 1.0, len(df)).clip(1, 8)
        if 'year' in df.columns:
            min_year, max_year = df['year'].min(), df['year'].max()
            denom = (max_year - min_year) if (max_year - min_year) != 0 else 1
            df['diagnostic_confidence'] = (df['year'] - min_year) / denom
            df['cumulative_risk_factor'] = (df['age'] - 65) * (df['year'] - min_year) / 10
        else:
            print("⚠️  No 'year' column found, creating synthetic year features")
            df['year'] = 2023
            df['diagnostic_confidence'] = 0.5
            df['cumulative_risk_factor'] = (df['age'] - 65) / 10
        df['age_at_diagnosis'] = df['age']
        alz_mask = df['alzheimers_diagnosis'] == 1
        noise = np.random.normal(0, 0.5, len(df))
        df.loc[alz_mask, 'diagnosis_year_uncertainty'] = noise[alz_mask]
        return df

    def _clean_and_impute_data(self, X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        if X_df.isna().values.any():
            num_cols = X_df.select_dtypes(include=['float64', 'int64']).columns
            if len(num_cols) > 0:
                med = X_df[num_cols].median()
                X_df[num_cols] = X_df[num_cols].fillna(med)
            non_num = X_df.select_dtypes(exclude=['float64', 'int64']).columns
            for col in non_num:
                if X_df[col].isna().any():
                    mode_val = X_df[col].mode().iloc[0] if len(X_df[col].mode()) > 0 else 0
                    X_df[col].fillna(mode_val, inplace=True)
        return X_df.values

    def _select_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                                 method: str = "f1_optimization") -> float:
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold, best_score = 0.5, 0.0
        if method == "f1_optimization":
            for th in tqdm(thresholds, desc="Optimizing threshold", unit="thresh", leave=False):
                f1 = f1_score(y_true, (y_proba >= th).astype(int))
                if f1 > best_score:
                    best_score, best_threshold = f1, th
        elif method == "precision_recall_balance":
            for th in tqdm(thresholds, desc="Optimizing threshold", unit="thresh", leave=False):
                preds = (y_proba >= th).astype(int)
                prec = precision_score(y_true, preds, zero_division=0)
                rec = recall_score(y_true, preds, zero_division=0)
                score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                if score > best_score:
                    best_score, best_threshold = score, th
        elif method == "cost_sensitive":
            fn_cost, fp_cost = 3.0, 1.0
            best_score = float("inf")
            for th in tqdm(thresholds, desc="Optimizing threshold", unit="thresh", leave=False):
                preds = (y_proba >= th).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
                total_cost = fp * fp_cost + fn * fn_cost
                if total_cost < best_score:
                    best_score, best_threshold = total_cost, th

        self.preprocessing_metadata['threshold_selection'] = {
            'method': method,
            'optimal_threshold': best_threshold,
            'best_score': best_score,
            'thresholds_evaluated': len(thresholds)
        }
        return best_threshold

    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        print(f"  🚀 Training Logistic Regression...")
        params = self.config.lr_params.copy()
        if self.config.handle_imbalance == "class_weight":
            params['class_weight'] = 'balanced'
        params.update({'solver': 'liblinear', 'max_iter': 500, 'tol': 1e-3,
                       'random_state': self.config.random_state})

        X_clean = self._clean_and_impute_data(X_train) if np.isnan(X_train).any() else X_train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        model = LogisticRegression(**params)
        model.fit(X_scaled, y_train)
        self.lr_scaler = scaler
        return model

    def _predict_model(self, model, model_name: str, X: np.ndarray) -> np.ndarray:
        if model_name == "logistic_regression" and self.lr_scaler is not None:
            X_clean = self._clean_and_impute_data(X) if np.isnan(X).any() else X
            X_scaled = self.lr_scaler.transform(X_clean)
            return model.predict(X_scaled)
        else:
            X_clean = self._clean_and_impute_data(X) if np.isnan(X).any() else X
            return model.predict(X_clean)

    def _predict_proba_model(self, model, model_name: str, X: np.ndarray) -> np.ndarray:
        try:
            if model_name == "logistic_regression" and self.lr_scaler is not None:
                X_clean = self._clean_and_impute_data(X) if np.isnan(X).any() else X
                X_scaled = self.lr_scaler.transform(X_clean)
                proba = model.predict_proba(X_scaled)
            else:
                X_clean = self._clean_and_impute_data(X) if np.isnan(X).any() else X
                proba = model.predict_proba(X_clean)

            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            if proba.shape[1] < 2:
                raise ValueError(f"Expected proba with >=2 columns, got {proba.shape}")
            return proba
        except Exception as e:
            logger.error(f"Error in _predict_proba_model for {model_name}: {e}")
            logger.error(f"Input shape: {X.shape}, Model type: {type(model)}")
            raise

    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> xgb.XGBClassifier:
        if getattr(self.config, 'enable_hyperparameter_tuning', False):
            return self._train_xgboost_with_tuning(X_train, y_train, X_val, y_val)
        else:
            print(f"  🚀 Training XGBoost (n_estimators={self.config.xgb_params.get('n_estimators', 50)})...")
            params = self.config.xgb_params.copy()
            if self.config.handle_imbalance == "class_weight":
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                params['scale_pos_weight'] = (neg_count / max(pos_count, 1))
            params.update({'early_stopping_rounds': 10, 'eval_metric': 'logloss', 'verbosity': 0})
            model = xgb.XGBClassifier(**params)
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train, verbose=False)
            return model

    def _train_xgboost_with_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
        def objective(trial):
            trial_start = time.time()
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10.0, log=True),
                'random_state': self.config.random_state,
                'eval_metric': 'logloss'
            }
            if self.config.handle_imbalance == "class_weight":
                neg = np.sum(y_train == 0); pos = np.sum(y_train == 1)
                params['scale_pos_weight'] = (neg / max(pos, 1))
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.config.random_state)
            scores = []
            for tr_idx, va_idx in cv.split(X_train, y_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]
                cv_params = params.copy()
                cv_params.update({'n_estimators': min(25, params['n_estimators']), 'tree_method': 'hist', 'verbosity': 0})
                model = xgb.XGBClassifier(**cv_params)
                model.fit(X_tr, y_tr, verbose=False)
                y_proba = model.predict_proba(X_va)[:, 1]
                scores.append(roc_auc_score(y_va, y_proba))
            if hasattr(self, 'tracker') and self.tracker:
                self.tracker["log"]({
                    'trial_number': trial.number,
                    'cv_roc_auc_mean': float(np.mean(scores)),
                    'cv_roc_auc_std': float(np.std(scores)),
                    'training_time': time.time() - trial_start
                })
            return float(np.mean(scores))

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.config.random_state))
        n_trials = getattr(self.config, 'n_trials', 20)
        print(f"  🔍 Hyperparameter tuning ({n_trials} trials)...")
        with tqdm(total=n_trials, desc="Optuna trials", unit="trial",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  ncols=60, ascii=True, position=1, leave=False) as pbar:
            def cb(study_, trial_):
                pbar.update(1)
                pbar.set_postfix({'best': f"{study_.best_value:.4f}"})
            study.optimize(objective, n_trials=n_trials, callbacks=[cb])

        best_params = study.best_params
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker["log"]({f"best_{k}": v for k, v in best_params.items()})
            self.tracker["log"]({"best_cv_score": float(study.best_value)})

        # Optional: importance plot saving skipped for brevity/robustness

        final_params = best_params.copy()
        final_params['random_state'] = self.config.random_state
        final_params['eval_metric'] = 'logloss'
        if self.config.handle_imbalance == "class_weight":
            neg = np.sum(y_train == 0); pos = np.sum(y_train == 1)
            final_params['scale_pos_weight'] = (neg / max(pos, 1))
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_train, y_train)
        return final_model

    def _evaluate_model(self, model, model_name: str, X: np.ndarray, y: np.ndarray, split_name: str) -> Dict[str, float]:
        y_pred = self._predict_model(model, model_name, X)
        y_pred_proba = self._predict_proba_model(model, model_name, X)[:, 1]
        return {
            f"{split_name}_accuracy": float(accuracy_score(y, y_pred)),
            f"{split_name}_precision": float(precision_score(y, y_pred, zero_division=0)),
            f"{split_name}_recall": float(recall_score(y, y_pred, zero_division=0)),
            f"{split_name}_f1": float(f1_score(y, y_pred)),
            f"{split_name}_roc_auc": float(roc_auc_score(y, y_pred_proba)) if len(np.unique(y)) == 2 else float('nan'),
            f"{split_name}_pr_auc": float(average_precision_score(y, y_pred_proba))
        }

    def _find_optimal_threshold(self, model, model_name: str, X_val: np.ndarray, y_val: np.ndarray) -> float:
        y_proba = self._predict_proba_model(model, model_name, X_val)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.1)
        best, best_f1 = 0.5, 0.0
        for th in thresholds:
            f1 = f1_score(y_val, (y_proba >= th).astype(int))
            if f1 > best_f1:
                best_f1, best = f1, th
        return float(best)

    def _save_plots(self, model_name: str, model, X_test: np.ndarray, y_test: np.ndarray, run_id: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = ensure_dir(Path("plots") / f"{model_name}_{timestamp}")
        try:
            y_pred = self._predict_model(model, model_name, X_test)
            y_proba = self._predict_proba_model(model, model_name, X_test)[:, 1]

            plt.style.use('default')
            sns.set_palette("husl")

            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["No Alzheimer's", "Alzheimer's"],
                        yticklabels=["No Alzheimer's", "Alzheimer's"])
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label'); plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(plots_dir / f'{model_name}_confusion_matrix.jpg', dpi=300, bbox_inches='tight')
            plt.close()

            # ROC & PR
            if len(np.unique(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_score = roc_auc_score(y_test, y_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, lw=2, label=f'AUC={auc_score:.3f}')
                plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
                plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{model_name} - ROC Curve')
                plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(plots_dir / f'{model_name}_roc_curve.jpg', dpi=300, bbox_inches='tight')
                plt.close()

                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = average_precision_score(y_test, y_proba)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, lw=2, label=f'AP={pr_auc:.3f}')
                plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{model_name} - PR Curve')
                plt.legend(loc="lower left"); plt.grid(alpha=0.3); plt.tight_layout()
                plt.savefig(plots_dir / f'{model_name}_pr_curve.jpg', dpi=300, bbox_inches='tight')
                plt.close()

            # Feature importance (XGB)
            if model_name == "xgboost" and hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                top = 20
                df_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(top)
                sns.barplot(data=df_imp, x='importance', y='feature')
                plt.title(f'{model_name} - Top {top} Feature Importance')
                plt.xlabel('Importance'); plt.ylabel('Feature'); plt.tight_layout()
                plt.savefig(plots_dir / f'{model_name}_feature_importance.jpg', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            logger.warning(f"Could not save plots for {model_name}: {e}")

    def train(self, run_type: str = "initial", tracker_type: str = "none") -> Dict[str, Any]:
        print("🚀 Starting model training pipeline...")
        run_name = self._generate_model_name(run_type, include_timestamp=True)
        self.tracker_enabled = tracker_type != "none"
        if self.tracker_enabled:
            print(f"✅ {tracker_type.upper()} tracking enabled")
        else:
            print("ℹ️  Using local JSON logging fallback")

        with tracker_run(run_name, params=vars(self.config)) as tr:
            self.tracker = tr
            return self._train_with_tracking(run_type)

    def _train_with_tracking(self, run_type: str) -> Dict[str, Any]:
        print("📊 Loading and preparing data...")
        with tqdm(total=4, desc="Data Pipeline", unit="step",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  ncols=80, ascii=True, position=0, leave=True) as pbar:

            pbar.set_description("Loading data")
            df = self._load_data(); pbar.update(1)

            pbar.set_description("Splitting data")
            train_df, val_df, test_df = self._patient_level_split(df); pbar.update(1)

            pbar.set_description("Preparing features")
            X_train, y_train, feature_names = self._prepare_features(train_df)
            X_val, y_val, _ = self._prepare_features(val_df)
            X_test, y_test, _ = self._prepare_features(test_df); pbar.update(1)

            pbar.set_description("Selecting features")
            selected_features = self._feature_selection(X_train, feature_names, y_train)
            self.feature_names = selected_features; pbar.update(1)

        total_samples = len(X_train) + len(X_val) + len(X_test)
        pos_rate = np.mean(y_train) * 100
        print(f"📊 Dataset: {total_samples:,} samples, {len(selected_features)} features")
        print(f"📈 Target: {np.bincount(y_train)[0]:,} negative, {np.bincount(y_train)[1]:,} positive ({pos_rate:.1f}% prevalence)\n")

        feat_idx = [feature_names.index(f) for f in selected_features]
        X_train_sel, X_val_sel, X_test_sel = X_train[:, feat_idx], X_val[:, feat_idx], X_test[:, feat_idx]
        X_train_bal, y_train_bal = self._handle_class_imbalance(X_train_sel, y_train)

        results: Dict[str, Any] = {}
        print("🤖 Training models...")
        for model_name in tqdm(self.config.models, desc="Training models", unit="model",
                               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                               ncols=80, ascii=True, position=0, dynamic_ncols=False,
                               mininterval=0.1, maxinterval=1.0, leave=True):
            if model_name == "logistic_regression":
                model = self._train_logistic_regression(X_train_bal, y_train_bal)
            elif model_name == "xgboost":
                model = self._train_xgboost(X_train_bal, y_train_bal, X_val_sel, y_val)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            train_metrics = self._evaluate_model(model, model_name, X_train_sel, y_train, "train")
            val_metrics   = self._evaluate_model(model, model_name, X_val_sel, y_val, "val")
            test_metrics  = self._evaluate_model(model, model_name, X_test_sel, y_test, "test")
            optimal_th = self._find_optimal_threshold(model, model_name, X_val_sel, y_val)

            model_metrics = {**train_metrics, **val_metrics, **test_metrics}
            model_metrics[f"{model_name}_optimal_threshold"] = float(optimal_th)

            if self.tracker_enabled and hasattr(self, 'tracker') and self.tracker:
                self.tracker["log"]({f"final_{model_name}_{k}": v for k, v in model_metrics.items()})

            # Log diagnostic plots to tracker, if enabled
            try:
                y_pred = self._predict_model(model, model_name, X_test_sel)
                log_plot({'probs': None, 'y_true': y_test, 'preds': y_pred,
                          'class_names': ["No Alzheimer's", "Alzheimer's"]},
                         f"{model_name}_confusion_matrix", "confusion_matrix") if self.tracker_enabled else None

                y_proba = self._predict_proba_model(model, model_name, X_test_sel)[:, 1]
                if self.tracker_enabled and len(np.unique(y_test)) == 2 and np.all((y_proba >= 0) & (y_proba <= 1)):
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    log_plot({'fpr': fpr, 'tpr': tpr}, f"{model_name}_roc_curve", "roc_curve")
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    log_plot({'precision': precision, 'recall': recall}, f"{model_name}_pr_curve", "pr_curve")
            except Exception as e:
                print(f"⚠️  Plot logging skipped for {model_name}: {e}")

            self.models[model_name] = model
            results[model_name] = {'metrics': model_metrics, 'optimal_threshold': float(optimal_th)}

        # Save local plots
        run_id_clean = self._generate_model_name(run_type, include_timestamp=False)
        for model_name, model in self.models.items():
            self._save_plots(model_name, model, X_test_sel, y_test, run_id_clean)

        # -----------------------------
        # Save REQUIRED serving artifacts
        # -----------------------------
        # Choose best model (prefer xgboost)
        best_name = "xgboost" if "xgboost" in self.models else "logistic_regression"
        best_model = self.models[best_name]
        best_res = results[best_name]
        optimal_threshold = float(best_res['optimal_threshold'])

        artifacts_dir = ensure_dir(Path("artifacts") / "latest", 0o777)

        # model.pkl
        write_pickle(artifacts_dir / "model.pkl", best_model)

        # feature_names.json (serving expects this name)
        write_json(artifacts_dir / "feature_names.json", self.feature_names)

        # threshold.json with key "optimal_threshold" (as run_serve.py loads)
        write_json(artifacts_dir / "threshold.json", {"optimal_threshold": optimal_threshold})

        # metrics.json — consolidate useful metrics + run id + model name
        # flatten metrics with JSON-friendly types
        metrics_payload: Dict[str, Any] = {
            "run_id": run_id_clean,
            "best_model": best_name,
            "feature_count": len(self.feature_names),
            "optimal_threshold": optimal_threshold,
            "timestamp": datetime.now().isoformat()
        }
        for mname, mres in results.items():
            for k, v in mres["metrics"].items():
                metrics_payload[f"{mname}_{k}"] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        write_json(artifacts_dir / "metrics.json", metrics_payload)

        # -----------------------------
        # Legacy files (back-compat)
        # -----------------------------
        write_json(artifacts_dir / "feature_list.json", self.feature_names)  # legacy alt
        model_meta = {
            "model_name": best_name,
            "created_at": datetime.now().isoformat(),
            "params": vars(self.config),
            "metrics": {"optimal_threshold": optimal_threshold, "feature_count": len(self.feature_names),
                        "model_type": best_name}
        }
        write_json(artifacts_dir / "model_meta.json", model_meta)

        # Also save full set under models/<run_id_clean>
        out_root = ensure_dir(Path(self.config.output_dir), 0o777)
        run_dir = ensure_dir(out_root / run_id_clean, 0o777)
        for mname, model in self.models.items():
            write_pickle(run_dir / f"{mname}.pkl", model)
            if mname == "logistic_regression" and self.lr_scaler is not None:
                write_pickle(run_dir / f"{mname}_scaler.pkl", self.lr_scaler)

        # Optional: tracker artifact upload
        if self.config.log_artifacts and self.tracker_enabled:
            try:
                log_artifact(run_dir, f"models-{run_id_clean}", "model")
            except Exception:
                pass

        # Feature importance logging
        if "xgboost" in self.models:
            xgb_model = self.models["xgboost"]
            fi_df = pd.DataFrame({'feature': self.feature_names,
                                  'importance': xgb_model.feature_importances_}).sort_values('importance', ascending=False)
            if self.tracker_enabled:
                try:
                    log_table(fi_df, "feature_importance")
                    if hasattr(self, 'tracker') and self.tracker:
                        self.tracker["log"]({
                            "model_n_trees": getattr(xgb_model, "n_estimators", None),
                            "model_max_depth": getattr(xgb_model, "max_depth", None),
                            "model_n_features": len(self.feature_names),
                            "model_feature_importance_mean": float(np.mean(xgb_model.feature_importances_)),
                            "model_feature_importance_std": float(np.std(xgb_model.feature_importances_)),
                        })
                except Exception:
                    pass

        print(f"🎉 Training complete! Model: {run_id_clean}")
        print(f"💾 Artifacts saved to: {artifacts_dir}")
        print(f"   - model.pkl\n   - feature_names.json\n   - threshold.json\n   - metrics.json\n")

        # WandB run id (optional)
        wandb_run_id = None
        if self.tracker_enabled and tracker_type == "wandb":
            try:
                import wandb
                if hasattr(wandb, 'run') and wandb.run:
                    wandb_run_id = wandb.run.id
            except Exception:
                pass

        return {
            'run_id': run_id_clean,
            'wandb_run_id': wandb_run_id,
            'artifact_path': str(run_dir),
            'latest_artifacts': [str(artifacts_dir / "model.pkl"),
                                 str(artifacts_dir / "feature_names.json"),
                                 str(artifacts_dir / "threshold.json"),
                                 str(artifacts_dir / "metrics.json")],
            'results': results
        }


def train(
    config_file: str = typer.Option("config/model.yaml", "--config", help="Configuration file path"),
    input_dir: Optional[str] = typer.Option(None, "--input-dir", help="Override input directory from config"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Override output directory from config"),
    max_features: Optional[int] = typer.Option(None, "--max-features", help="Override max features from config"),
    handle_imbalance: Optional[str] = typer.Option(None, "--handle-imbalance", help="Override imbalance handling from config"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb-project", help="Override wandb project from config"),
    wandb_entity: Optional[str] = typer.Option(None, "--wandb-entity", help="Override wandb entity from config"),
    run_type: str = typer.Option("initial", "--run-type", help="Run type (initial, balanced, feature_optimized, hyperparameter_tuned, final, production)"),
    tracker: Optional[str] = typer.Option(None, "--tracker"),
) -> None:
    """
    Train machine learning models for Alzheimer's prediction.
    """
    from src.config import load_config
    config = load_config("model", config_file)

    if input_dir is not None:
        config.input_dir = input_dir
    if output_dir is not None:
        config.output_dir = output_dir
    if max_features is not None:
        config.max_features = max_features
    if handle_imbalance is not None:
        config.handle_imbalance = handle_imbalance
    if wandb_project is not None:
        config.wandb_project = wandb_project
    if wandb_entity is not None:
        config.wandb_entity = wandb_entity

    # Tracking setup
    if tracker is None:
        from utils import setup_experiment_tracker
        global_tracker, chosen_tracker_type = setup_experiment_tracker()
    else:
        tracker_lower = tracker.lower()
        print(f"🔬 Setting up experiment tracking: {tracker_lower}")
        if tracker_lower == "wandb":
            os.environ['NON_INTERACTIVE'] = 'true'
            from utils import setup_wandb
            global_tracker, chosen_tracker_type = setup_wandb()
        elif tracker_lower == "mlflow":
            os.environ['NON_INTERACTIVE'] = 'true'
            from utils import setup_mlflow
            global_tracker, chosen_tracker_type = setup_mlflow()
        elif tracker_lower == "none":
            os.environ['NON_INTERACTIVE'] = 'true'
            global_tracker, chosen_tracker_type = None, "none"
        else:
            print(f"❌ Invalid tracker option: {tracker}")
            print("Valid options: none, mlflow, wandb")
            return

    # Ensure artifacts dir root exists with wide permissions (useful in containers)
    ensure_dir(Path("artifacts"), 0o777)

    trainer = ModelTrainer(config)
    results = trainer.train(run_type, chosen_tracker_type)

    print("🎉 Training completed successfully!")
    print(f"🆔 Run ID: {results['run_id']}")
    print(f"📁 Artifact path: {results['artifact_path']}")
    if results.get('wandb_run_id'):
        try:
            import wandb
            if hasattr(wandb, 'run') and wandb.run:
                print(f"🔗 View run online: {wandb.run.get_url()}")
        except ImportError:
            pass


if __name__ == "__main__":
    typer.run(train)
