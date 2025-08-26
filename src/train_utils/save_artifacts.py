"""
Artifact saving utilities for the Alzearly training pipeline.
Ensures consistent saving of model, feature names, threshold, and metrics.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


def _json_serializable(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(item) for item in obj]
    else:
        return obj

from src.io.paths import (
    get_latest_artifacts_dir,
    get_model_path,
    get_feature_names_path,
    get_threshold_path,
    get_metrics_path,
)


def save_model(model: Any, model_name: str = "model.pkl") -> Path:
    """Save trained model to artifacts/latest/."""
    model_path = get_model_path(model_name)
    
    print(f"Saving model to: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model_path


def save_feature_names(feature_names: List[str]) -> Path:
    """Save feature names to artifacts/latest/feature_names.json."""
    feature_path = get_feature_names_path()
    
    print(f"Saving feature names to: {feature_path}")
    
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    return feature_path


def save_threshold(threshold: float) -> Path:
    """Save optimal threshold to artifacts/latest/threshold.json."""
    threshold_path = get_threshold_path()
    
    print(f"Saving threshold to: {threshold_path}")
    
    threshold_data = {"threshold": threshold}
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    
    return threshold_path


def save_metrics(metrics: Dict[str, Any]) -> Path:
    """Save training metrics to artifacts/latest/metrics.json."""
    metrics_path = get_metrics_path()
    
    print(f"Saving metrics to: {metrics_path}")
    
    # Convert numpy types to JSON serializable types
    serializable_metrics = _json_serializable(metrics)
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    return metrics_path


def save_all_artifacts(
    model: Any,
    feature_names: List[str],
    threshold: float,
    metrics: Dict[str, Any],
    model_name: str = "model.pkl"
) -> Dict[str, Path]:
    """Save all required artifacts after training."""
    
    print("Saving training artifacts...")
    
    # Ensure artifacts directory exists
    get_latest_artifacts_dir()
    
    # Save all artifacts
    saved_paths = {
        "model": save_model(model, model_name),
        "feature_names": save_feature_names(feature_names),
        "threshold": save_threshold(threshold),
        "metrics": save_metrics(metrics)
    }
    
    print("All artifacts saved successfully!")
    
    return saved_paths


def load_model(model_name: str = "model.pkl") -> Any:
    """Load trained model from artifacts/latest/."""
    model_path = get_model_path(model_name)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_feature_names() -> List[str]:
    """Load feature names from artifacts/latest/feature_names.json."""
    feature_path = get_feature_names_path()
    
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature names not found at: {feature_path}")
    
    with open(feature_path, 'r') as f:
        return json.load(f)


def load_threshold() -> float:
    """Load threshold from artifacts/latest/threshold.json."""
    threshold_path = get_threshold_path()
    
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold not found at: {threshold_path}")
    
    with open(threshold_path, 'r') as f:
        data = json.load(f)
        return data["threshold"]


def load_metrics() -> Dict[str, Any]:
    """Load metrics from artifacts/latest/metrics.json."""
    metrics_path = get_metrics_path()
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found at: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)
