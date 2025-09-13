"""
Model evaluation module for the ML project.

Computes comprehensive metrics including AUROC, AUPRC, F1 across thresholds,
chooses optimal thresholds, and evaluates with time windows.
"""

import logging
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with threshold optimization and time windows."""
    
    def __init__(self, model_path, data_path, output_dir="artifacts"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        self.model = self._load_model()
        self.data = self._load_data()
        
        # Threshold evaluation parameters
        self.thresholds = np.arange(0.1, 0.9, 0.05)
        self.recall_target = 0.8  # Fallback threshold target
        
    def _load_model(self):
        """Load trained model from pickle file."""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded model: {self.model_path.name}")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def _load_data(self):
        """Load evaluation data."""
        try:
            # Try different file formats
            if self.data_path.suffix == '.parquet':
                data = pd.read_parquet(self.data_path)
            elif self.data_path.suffix == '.csv':
                data = pd.read_csv(self.data_path)
            else:
                # Assume it's a directory with parquet files
                data = pd.read_parquet(self.data_path)
            
            print(f"Loaded data: {data.shape[0]:,} samples, {data.shape[1]} features")
            return data
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame):
        """Prepare features and target for evaluation."""
        # Load the model's metadata to get the exact feature names used during training
        model_dir = self.model_path.parent
        metadata_file = model_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            feature_names = metadata.get('feature_names', [])
            print(f"Using {len(feature_names)} features from model metadata")
        else:
            # Fallback: exclude non-feature columns
            exclude_cols = ['patient_id', 'year', 'alzheimers_diagnosis']
            feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"⚠️  No metadata found, using {len(feature_names)} available features")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Convert categorical columns to numeric (if they haven't been encoded yet)
        # Note: In the new preprocessing pipeline, categorical columns are already encoded
        # This is a fallback for backward compatibility
        categorical_cols = ['sex', 'region', 'occupation', 'education_level', 'marital_status', 'insurance_type']
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes
        
        # Use only the features that were used during training
        available_features = [col for col in feature_names if col in df.columns]
        missing_features = [col for col in feature_names if col not in df.columns]
        
        if missing_features:
            print(f"⚠️  Missing {len(missing_features)} features")
        
        # Prepare features and target
        X = df[available_features].values.astype(float)
        y = df['alzheimers_diagnosis'].values
        
        print(f"Prepared: {X.shape[0]:,} samples, {X.shape[1]} features")
        return X, y, available_features
    
    def _compute_metrics_at_threshold(self, y_true, y_pred_proba, 
                                    threshold):
        """Compute metrics at a specific threshold."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        return {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold using max F1 with fallback to recall target."""
        threshold_metrics = []
        
        for threshold in self.thresholds:
            metrics = self._compute_metrics_at_threshold(y_true, y_pred_proba, threshold)
            threshold_metrics.append(metrics)
        
        # Find threshold with maximum F1
        max_f1_idx = np.argmax([m['f1'] for m in threshold_metrics])
        max_f1_threshold = threshold_metrics[max_f1_idx]['threshold']
        max_f1_score = threshold_metrics[max_f1_idx]['f1']
        
        # Find threshold with recall >= target (fallback)
        recall_target_idx = None
        for i, metrics in enumerate(threshold_metrics):
            if metrics['recall'] >= self.recall_target:
                recall_target_idx = i
                break
        
        if recall_target_idx is not None:
            fallback_threshold = threshold_metrics[recall_target_idx]['threshold']
            fallback_recall = threshold_metrics[recall_target_idx]['recall']
        else:
            fallback_threshold = max_f1_threshold
            fallback_recall = threshold_metrics[max_f1_idx]['recall']
        
        return {
            'optimal_threshold': max_f1_threshold,
            'optimal_f1': max_f1_score,
            'fallback_threshold': fallback_threshold,
            'fallback_recall': fallback_recall,
            'threshold_metrics': threshold_metrics
        }
    
    def _evaluate_time_window(self, df, y_pred_proba, 
                            threshold, window_years=1):
        """Evaluate metrics with ±N-year window for positive cases."""
        df_eval = df.copy()
        df_eval['prediction'] = (y_pred_proba >= threshold).astype(int)
        df_eval['prediction_proba'] = y_pred_proba
        
        # Group by patient and find positive predictions
        positive_predictions = df_eval[df_eval['prediction'] == 1].copy()
        
        if len(positive_predictions) == 0:
            return {
                'window_accuracy': 0.0,
                'window_precision': 0.0,
                'window_recall': 0.0,
                'window_f1': 0.0,
                'window_positive_patients': 0
            }
        
        # Check if we have time-series data with year column
        if 'year' not in df_eval.columns:
            logger.warning("No 'year' column found in data. Skipping time window evaluation.")
            return {
                'window_accuracy': 0.0,
                'window_precision': 0.0,
                'window_recall': 0.0,
                'window_f1': 0.0,
                'window_positive_patients': len(positive_predictions)
            }
        
        # For each positive prediction, check if there's a true positive within ±window_years
        window_correct = 0
        total_positive_preds = len(positive_predictions)
        
        for _, pred_row in tqdm(positive_predictions.iterrows(), 
                               total=len(positive_predictions), 
                               desc="Evaluating time windows", 
                               unit="prediction",
                               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                               ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                               mininterval=0.1, maxinterval=1.0):
            patient_id = pred_row['patient_id']
            pred_year = pred_row['year']
            
            # Find all records for this patient
            patient_records = df_eval[df_eval['patient_id'] == patient_id]
            
            # Check if there's a true positive within the window
            window_start = pred_year - window_years
            window_end = pred_year + window_years
            
            window_records = patient_records[
                (patient_records['year'] >= window_start) & 
                (patient_records['year'] <= window_end)
            ]
            
            if any(window_records['alzheimers_diagnosis'] == 1):
                window_correct += 1
        
        window_accuracy = window_correct / total_positive_preds if total_positive_preds > 0 else 0.0
        
        # Calculate other metrics based on window accuracy
        # This is a simplified approach - in practice you might want more sophisticated window metrics
        return {
            'window_accuracy': window_accuracy,
            'window_precision': window_accuracy,  # Simplified
            'window_recall': window_accuracy,     # Simplified
            'window_f1': window_accuracy,         # Simplified
            'window_positive_patients': total_positive_preds
        }
    
    def _save_plots(self, y_true, y_pred_proba, threshold_metrics, run_id):
        """Save evaluation plots with meaningful names."""
        # Create plots directory with descriptive name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plots_dir = Path("plots") / f"evaluation_{timestamp}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Evaluation - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'evaluation_roc_curve.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Evaluation - Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'evaluation_pr_curve.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Threshold Analysis
        plt.figure(figsize=(12, 8))
        thresholds = [m['threshold'] for m in threshold_metrics]
        metrics_df = pd.DataFrame(threshold_metrics)
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, metrics_df['accuracy'], 'b-', label='Accuracy')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, metrics_df['precision'], 'g-', label='Precision')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, metrics_df['recall'], 'r-', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(thresholds, metrics_df['f1'], 'm-', label='F1')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'evaluation_threshold_analysis.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Saved evaluation plots
    
    def evaluate(self):
        """Run comprehensive evaluation."""
        # Starting comprehensive model evaluation
        print("Starting comprehensive model evaluation...")
        
        # Initialize wandb if not already initialized
        print("Initializing Weights & Biases...")
        if not wandb.run:
            run_name = f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="alz_detect",
                name=run_name,
                config={
                    "model_path": str(self.model_path),
                    "data_path": str(self.data_path),
                    "output_dir": str(self.output_dir)
                }
            )
        
        # Prepare data
        X, y_true, feature_names = self._prepare_features(self.data)
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Compute basic metrics
        auroc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        
        # Find optimal thresholds
        threshold_analysis = self._find_optimal_threshold(y_true, y_pred_proba)
        
        # Compute metrics at optimal threshold
        optimal_metrics = self._compute_metrics_at_threshold(
            y_true, y_pred_proba, threshold_analysis['optimal_threshold']
        )
        
        # Evaluate with time window
        window_metrics = self._evaluate_time_window(
            self.data, y_pred_proba, threshold_analysis['optimal_threshold']
        )
        
        # Prepare results
        results = {
            'basic_metrics': {
                'auroc': auroc,
                'auprc': auprc
            },
            'threshold_analysis': threshold_analysis,
            'optimal_metrics': optimal_metrics,
            'window_metrics': window_metrics,
            'data_info': {
                'n_samples': len(y_true),
                'n_features': len(feature_names),
                'positive_rate': np.mean(y_true),
                'feature_names': feature_names
            }
        }
        
        # Log to wandb
        wandb.log({
            'evaluation_auroc': auroc,
            'evaluation_auprc': auprc,
            'evaluation_optimal_threshold': threshold_analysis['optimal_threshold'],
            'evaluation_optimal_f1': threshold_analysis['optimal_f1'],
            'evaluation_fallback_threshold': threshold_analysis['fallback_threshold'],
            'evaluation_fallback_recall': threshold_analysis['fallback_recall'],
            'evaluation_accuracy': optimal_metrics['accuracy'],
            'evaluation_precision': optimal_metrics['precision'],
            'evaluation_recall': optimal_metrics['recall'],
            'evaluation_f1': optimal_metrics['f1'],
            'evaluation_window_accuracy': window_metrics['window_accuracy'],
            'evaluation_window_positive_patients': window_metrics['window_positive_patients']
        })
        
        # Save threshold analysis
        threshold_file = self.output_dir / "threshold.json"
        with open(threshold_file, 'w') as f:
            json.dump({
                'optimal_threshold': threshold_analysis['optimal_threshold'],
                'optimal_f1': threshold_analysis['optimal_f1'],
                'fallback_threshold': threshold_analysis['fallback_threshold'],
                'fallback_recall': threshold_analysis['fallback_recall'],
                'threshold_metrics': threshold_analysis['threshold_metrics']
            }, f, indent=2)
        
        # Save plots
        run_id = wandb.run.id if wandb.run else "evaluation"
        self._save_plots(y_true, y_pred_proba, threshold_analysis['threshold_metrics'], run_id)
        
        print(f"Evaluation complete! Results saved to {self.output_dir}")
        print(f"AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
        print(f"Optimal threshold: {threshold_analysis['optimal_threshold']:.3f} (F1: {threshold_analysis['optimal_f1']:.3f})")
        
        return results


def evaluate_model(
    model_path,
    data_path,
    output_dir="artifacts"
):
    """Evaluate a trained model with comprehensive metrics."""
    evaluator = ModelEvaluator(model_path, data_path, output_dir)
    results = evaluator.evaluate()
    
    print("Model evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def evaluate(
        model_path = typer.Argument(..., help="Path to trained model (.pkl)"),
        data_path = typer.Argument(..., help="Path to evaluation data"),
        output_dir = typer.Option("artifacts", help="Output directory for results")
    ):
        """Evaluate a trained model with comprehensive metrics."""
        evaluate_model(model_path, data_path, output_dir)
    
    app()
