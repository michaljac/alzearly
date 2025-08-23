"""
Model training module for Alzheimer's prediction.

Handles data loading, patient-level splits, class imbalance, feature selection,
and training of multiple models with wandb logging.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pickle

import typer
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import wandb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data configuration
    input_dir: str = "data/featurized"
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
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    })
    
    # Logistic Regression parameters
    lr_params: Dict[str, Any] = field(default_factory=lambda: {
        "random_state": 42,
        "max_iter": 1000
    })
    
    # Output configuration
    output_dir: str = "models"
    save_metadata: bool = True
    
    # Wandb configuration
    wandb_project: str = "alzheimers-prediction"
    wandb_entity: Optional[str] = None
    log_artifacts: bool = True


class ModelTrainer:
    """Handles model training with patient-level splits and feature selection."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.feature_names = []
        self.preprocessing_metadata = {}
        
    def _load_data(self) -> pd.DataFrame:
        """Load featurized data from partitioned Parquet files."""
        logger.info(f"Loading data from {self.config.input_dir}")
        
        # Use Polars to read partitioned data efficiently
        input_path = Path(self.config.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_path} does not exist")
        
        # Read all parquet files recursively
        df = pl.scan_parquet(str(input_path / "**" / "*.parquet")).collect()
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Convert to pandas for sklearn compatibility
        df_pandas = df.to_pandas()
        
        return df_pandas
    
    def _patient_level_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data at patient level to avoid temporal leakage."""
        logger.info("Performing patient-level split...")
        
        # Get unique patients
        patients = df['patient_id'].unique()
        logger.info(f"Total unique patients: {len(patients)}")
        
        # Split patients (not individual rows)
        if self.config.stratify:
            # Calculate patient-level target (majority class per patient)
            patient_targets = df.groupby('patient_id')[self.config.target_column].agg(
                lambda x: 1 if x.sum() > len(x) / 2 else 0
            )
            
            # Check if we have enough samples for stratification
            target_counts = patient_targets.value_counts()
            min_samples_per_class = 2
            
            if len(target_counts) < 2 or target_counts.min() < min_samples_per_class:
                logger.warning(f"Insufficient samples for stratification. Using random split. Target counts: {target_counts.to_dict()}")
                stratify = False
            else:
                stratify = True
        else:
            stratify = False
        
        if stratify:
            # Split patients with stratification
            train_patients, test_patients = train_test_split(
                patients,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=patient_targets.loc[patients]
            )
            
            # Further split training patients for validation
            train_patients, val_patients = train_test_split(
                train_patients,
                test_size=self.config.val_size,
                random_state=self.config.random_state,
                stratify=patient_targets.loc[train_patients]
            )
        else:
            # Simple random split
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
        
        # Filter data by patient splits
        train_df = df[df['patient_id'].isin(train_patients)]
        val_df = df[df['patient_id'].isin(val_patients)]
        test_df = df[df['patient_id'].isin(test_patients)]
        
        logger.info(f"Train: {len(train_df)} rows ({len(train_patients)} patients)")
        logger.info(f"Val: {len(val_df)} rows ({len(val_patients)} patients)")
        logger.info(f"Test: {len(test_df)} rows ({len(test_patients)} patients)")
        
        return train_df, val_df, test_df
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target, excluding specified columns."""
        logger.info("Preparing features and target...")
        
        # Exclude specified columns
        feature_cols = [col for col in df.columns if col not in self.config.exclude_columns + [self.config.target_column]]
        
        # Separate numeric and categorical columns
        numeric_cols = []
        categorical_cols = []
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        logger.info(f"Numeric columns: {len(numeric_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        
        # Handle categorical columns with one-hot encoding
        if categorical_cols:
            logger.info("Encoding categorical features...")
            # Use pandas get_dummies for one-hot encoding (without prefix to avoid length mismatch)
            categorical_df = pd.get_dummies(df[categorical_cols])
            numeric_df = df[numeric_cols] if numeric_cols else pd.DataFrame()
            
            # Combine numeric and encoded categorical features
            if not numeric_df.empty:
                X = pd.concat([numeric_df, categorical_df], axis=1)
            else:
                X = categorical_df
            
            # Update feature names to include encoded columns
            feature_cols = list(X.columns)
        else:
            X = df[feature_cols]
        
        # Convert to numpy arrays
        X_array = X.values.astype(float)
        y_array = df[self.config.target_column].values
        
        logger.info(f"Final features shape: {X_array.shape}")
        logger.info(f"Target distribution: {np.bincount(y_array)}")
        
        return X_array, y_array, feature_cols
    
    def _feature_selection(self, X_train: np.ndarray, feature_names: List[str], y_train: np.ndarray) -> List[str]:
        """Perform feature selection using variance threshold and XGBoost importance."""
        logger.info("Performing feature selection...")
        
        # Step 1: Variance threshold
        variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_train_var = variance_selector.fit_transform(X_train)
        var_features = [feature_names[i] for i in variance_selector.get_support(indices=True)]
        
        logger.info(f"After variance threshold: {len(var_features)} features")
        
        # Step 2: XGBoost importance-based selection
        if len(var_features) > self.config.max_features:
            logger.info(f"Selecting top {self.config.max_features} features by XGBoost importance...")
            
            # Train a quick XGBoost model to get feature importance
            xgb_selector = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.config.random_state,
                eval_metric='logloss'
            )
            
            # Get indices of variance-selected features
            var_indices = [feature_names.index(f) for f in var_features]
            X_train_var_subset = X_train[:, var_indices]
            
            # Fit XGBoost
            xgb_selector.fit(X_train_var_subset, y_train)
            
            # Get feature importance
            importance_scores = xgb_selector.feature_importances_
            feature_importance = list(zip(var_features, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_features = [f[0] for f in feature_importance[:self.config.max_features]]
            
            logger.info(f"Selected {len(selected_features)} features by importance")
            
            # Store preprocessing metadata
            self.preprocessing_metadata['feature_selection'] = {
                'variance_threshold': self.config.variance_threshold,
                'max_features': self.config.max_features,
                'variance_selected': len(var_features),
                'final_selected': len(selected_features),
                'feature_importance': dict(feature_importance[:self.config.max_features])
            }
            
            return selected_features
        else:
            logger.info(f"Using all {len(var_features)} features after variance threshold")
            self.preprocessing_metadata['feature_selection'] = {
                'variance_threshold': self.config.variance_threshold,
                'max_features': len(var_features),
                'variance_selected': len(var_features),
                'final_selected': len(var_features)
            }
            return var_features
    
    def _handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance using specified method."""
        logger.info(f"Handling class imbalance with method: {self.config.handle_imbalance}")
        
        if self.config.handle_imbalance == "smote":
            smote = SMOTE(random_state=self.config.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"After SMOTE: {np.bincount(y_resampled)}")
            return X_resampled, y_resampled
        elif self.config.handle_imbalance == "class_weight":
            # Will be handled in model training
            logger.info("Using class_weight in model training")
            return X, y
        else:
            logger.info("No class imbalance handling")
            return X, y
    
    def _train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train logistic regression model."""
        logger.info("Training Logistic Regression...")
        
        params = self.config.lr_params.copy()
        if self.config.handle_imbalance == "class_weight":
            params['class_weight'] = 'balanced'
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> xgb.XGBClassifier:
        """Train XGBoost model with optional hyperparameter tuning."""
        logger.info("Training XGBoost...")
        
        if hasattr(self.config, 'enable_hyperparameter_tuning') and self.config.enable_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning with Optuna...")
            return self._train_xgboost_with_tuning(X_train, y_train, X_val, y_val)
        else:
            # Use default parameters
            params = self.config.xgb_params.copy()
            if self.config.handle_imbalance == "class_weight":
                # Calculate scale_pos_weight
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                params['scale_pos_weight'] = neg_count / pos_count
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            return model
    
    def _train_xgboost_with_tuning(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost with hyperparameter tuning using Optuna."""
        
        def objective(trial):
            trial_start_time = time.time()
            
            # Define hyperparameter search space
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
            
            # Handle class imbalance
            if self.config.handle_imbalance == "class_weight":
                neg_count = np.sum(y_train == 0)
                pos_count = np.sum(y_train == 1)
                params['scale_pos_weight'] = neg_count / pos_count
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_cv_train, y_cv_train)
                
                y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
                score = roc_auc_score(y_cv_val, y_pred_proba)
                scores.append(score)
            
            # Log trial results to wandb
            wandb.log({
                'step': trial.number,  # Use 'step' for better graph visualization
                'trial_number': trial.number,
                'cv_roc_auc_mean': np.mean(scores),
                'cv_roc_auc_std': np.std(scores),
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'learning_rate': params['learning_rate'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'training_time': time.time() - trial_start_time
            })
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        n_trials = getattr(self.config, 'n_trials', 50)
        study.optimize(objective, n_trials=n_trials)
        
        # Log best parameters
        best_params = study.best_params
        wandb.log({
            'best_n_estimators': best_params['n_estimators'],
            'best_max_depth': best_params['max_depth'],
            'best_learning_rate': best_params['learning_rate'],
            'best_subsample': best_params['subsample'],
            'best_colsample_bytree': best_params['colsample_bytree'],
            'best_reg_alpha': best_params['reg_alpha'],
            'best_reg_lambda': best_params['reg_lambda'],
            'best_cv_score': study.best_value
        })
        
        # Log parameter importance plots
        try:
            # Parameter importance
            importance = optuna.importance.get_param_importances(study)
            if importance:
                importance_df = pd.DataFrame(list(importance.items()), columns=['parameter', 'importance'])
                wandb.log({"parameter_importance": wandb.Table(dataframe=importance_df)})
                
                # Log individual parameter importance values
                for param, imp in importance.items():
                    wandb.log({f"param_importance_{param}": imp})
                
                # Save parameter importance plot
                plots_dir = Path("plots") / wandb.run.id
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                plt.figure(figsize=(10, 6))
                importance_df = importance_df.sort_values('importance', ascending=True)
                plt.barh(importance_df['parameter'], importance_df['importance'])
                plt.xlabel('Importance')
                plt.title('Hyperparameter Importance')
                plt.tight_layout()
                plt.savefig(plots_dir / 'hyperparameter_importance.jpg', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not create parameter importance plots: {e}")
        
        # Train final model with best parameters
        final_params = best_params.copy()
        final_params['random_state'] = self.config.random_state
        final_params['eval_metric'] = 'logloss'
        
        if self.config.handle_imbalance == "class_weight":
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            final_params['scale_pos_weight'] = neg_count / pos_count
        
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_train, y_train)
        
        return final_model
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, split_name: str) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            f"{split_name}_accuracy": accuracy_score(y, y_pred),
            f"{split_name}_precision": precision_score(y, y_pred),
            f"{split_name}_recall": recall_score(y, y_pred),
            f"{split_name}_f1": f1_score(y, y_pred),
            f"{split_name}_roc_auc": roc_auc_score(y, y_pred_proba),
            f"{split_name}_pr_auc": average_precision_score(y, y_pred_proba),
        }
        
        return metrics
    
    def _find_optimal_threshold(self, model, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Find optimal classification threshold using validation set."""
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def _save_plots(self, model_name: str, model, X_test: np.ndarray, y_test: np.ndarray, run_id: str):
        """Save important plots as JPG files in plots directory."""
        plots_dir = Path("plots") / run_id
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Set style for better looking plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=["No Alzheimer's", "Alzheimer's"],
                       yticklabels=["No Alzheimer's", "Alzheimer's"])
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(plots_dir / f'{model_name}_confusion_matrix.jpg', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. ROC Curve
            if len(np.unique(y_test)) == 2:
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{model_name} - ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / f'{model_name}_roc_curve.jpg', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Precision-Recall Curve
                plt.figure(figsize=(8, 6))
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = average_precision_score(y_test, y_pred_proba)
                
                plt.plot(recall, precision, color='blue', lw=2, 
                        label=f'PR curve (AP = {pr_auc:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{model_name} - Precision-Recall Curve')
                plt.legend(loc="lower left")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / f'{model_name}_pr_curve.jpg', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Feature Importance (for XGBoost)
            if model_name == "xgboost" and hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                # Get top 20 features
                top_features = 20
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(top_features)
                
                sns.barplot(data=feature_importance, x='importance', y='feature')
                plt.title(f'{model_name} - Top {top_features} Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(plots_dir / f'{model_name}_feature_importance.jpg', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Saved plots for {model_name} to {plots_dir}")
            
        except Exception as e:
            logger.warning(f"Could not save plots for {model_name}: {e}")
    
    def train(self) -> Dict[str, Any]:
        """Main training pipeline."""
        logger.info("Starting model training pipeline...")
        
        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=vars(self.config)
        )
        
        # Load data
        df = self._load_data()
        
        # Patient-level split
        train_df, val_df, test_df = self._patient_level_split(df)
        
        # Prepare features
        X_train, y_train, feature_names = self._prepare_features(train_df)
        X_val, y_val, _ = self._prepare_features(val_df)
        X_test, y_test, _ = self._prepare_features(test_df)
        
        # Feature selection
        selected_features = self._feature_selection(X_train, feature_names, y_train)
        self.feature_names = selected_features
        
        # Get indices of selected features
        feature_indices = [feature_names.index(f) for f in selected_features]
        
        # Apply feature selection to all datasets
        X_train_selected = X_train[:, feature_indices]
        X_val_selected = X_val[:, feature_indices]
        X_test_selected = X_test[:, feature_indices]
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self._handle_class_imbalance(X_train_selected, y_train)
        
        # Train models
        results = {}
        
        for model_name in self.config.models:
            logger.info(f"Training {model_name}...")
            
            if model_name == "logistic_regression":
                model = self._train_logistic_regression(X_train_balanced, y_train_balanced)
            elif model_name == "xgboost":
                model = self._train_xgboost(X_train_balanced, y_train_balanced, X_val_selected, y_val)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Evaluate model
            train_metrics = self._evaluate_model(model, X_train_selected, y_train, "train")
            val_metrics = self._evaluate_model(model, X_val_selected, y_val, "val")
            test_metrics = self._evaluate_model(model, X_test_selected, y_test, "test")
            
            # Find optimal threshold
            optimal_threshold = self._find_optimal_threshold(model, X_val_selected, y_val)
            
            # Combine metrics
            model_metrics = {**train_metrics, **val_metrics, **test_metrics}
            model_metrics[f"{model_name}_optimal_threshold"] = optimal_threshold
            
                    # Log to wandb (use different keys to avoid overwriting hyperparameter tuning data)
            final_metrics = {}
            for k, v in model_metrics.items():
                final_metrics[f"final_{model_name}_{k}"] = v
            wandb.log(final_metrics)
            
            # Log confusion matrix
            y_pred = model.predict(X_test_selected)
            cm = confusion_matrix(y_test, y_pred)
            wandb.log({f"{model_name}_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, 
                y_true=y_test, 
                preds=y_pred,
                class_names=["No Alzheimer's", "Alzheimer's"]
            )})
            
            # Log ROC and PR curves with error handling
            try:
                y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
                
                # Ensure we have both classes and reasonable probabilities
                if len(np.unique(y_test)) == 2 and np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
                    # ROC curve
                    wandb.log({f"{model_name}_roc_curve": wandb.plot.roc_curve(
                        y_test, 
                        y_pred_proba,
                        labels=["No Alzheimer's", "Alzheimer's"]
                    )})
                    
                    # PR curve
                    wandb.log({f"{model_name}_pr_curve": wandb.plot.pr_curve(
                        y_test, 
                        y_pred_proba,
                        labels=["No Alzheimer's", "Alzheimer's"]
                    )})
                else:
                    logger.warning(f"Skipping ROC/PR curves for {model_name} - insufficient class diversity or invalid probabilities")
            except Exception as e:
                logger.warning(f"Could not create ROC/PR curves for {model_name}: {e}")
            
            # Store model and results
            self.models[model_name] = model
            results[model_name] = {
                'metrics': model_metrics,
                'optimal_threshold': optimal_threshold
            }
            
            logger.info(f"{model_name} - Val ROC AUC: {val_metrics['val_roc_auc']:.3f}")
        
        # Save plots for each model
        run_id = wandb.run.id
        for model_name, model in self.models.items():
            self._save_plots(model_name, model, X_test_selected, y_test, run_id)
        
        # Save models and metadata
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        run_id = wandb.run.id
        artifact_path = output_path / run_id
        artifact_path.mkdir(exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_file = artifact_path / f"{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        if self.config.save_metadata:
            metadata = {
                'run_id': run_id,
                'feature_names': self.feature_names,
                'preprocessing_metadata': self.preprocessing_metadata,
                'results': results,
                'config': vars(self.config)
            }
            
            metadata_file = artifact_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        # Log artifacts to wandb
        if self.config.log_artifacts:
            artifact = wandb.Artifact(
                name=f"models-{run_id}",
                type="model",
                description="Trained models and metadata"
            )
            artifact.add_dir(str(artifact_path))
            wandb.log_artifact(artifact)
        
        # Log feature importance for XGBoost
        if "xgboost" in self.models:
            xgb_model = self.models["xgboost"]
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            wandb.log({"feature_importance": wandb.Table(dataframe=feature_importance)})
            
            # Log model complexity metrics
            wandb.log({
                "model_n_trees": xgb_model.n_estimators,
                "model_max_depth": xgb_model.max_depth,
                "model_n_features": len(self.feature_names),
                "model_feature_importance_mean": np.mean(xgb_model.feature_importances_),
                "model_feature_importance_std": np.std(xgb_model.feature_importances_)
            })
        
        wandb.finish()
        
        logger.info(f"Training complete! Run ID: {run_id}")
        logger.info(f"Artifacts saved to: {artifact_path}")
        
        return {
            'run_id': run_id,
            'artifact_path': str(artifact_path),
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
) -> None:
    """
    Train machine learning models for Alzheimer's prediction.
    
    Example:
        python cli.py train --config config/model.yaml
        python cli.py train --max-features 100 --handle-imbalance smote
    """
    # Load configuration from file
    from src.config import load_config
    config = load_config("model", config_file)
    
    # Override config values if provided as command line arguments
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
    
    # Create trainer and run training
    trainer = ModelTrainer(config)
    results = trainer.train()
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Run ID: {results['run_id']}")
    logger.info(f"Artifact path: {results['artifact_path']}")


if __name__ == "__main__":
    typer.run(train)
