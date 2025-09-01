"""
Artifact saving utilities for the Alzearly training pipeline.
Ensures consistent saving of model, feature names, threshold, and metrics.
Creates directories if missing and sets permissive permissions.
"""

import json
import pickle
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

# Permissions to apply (dirs: rwx for all; files: rw for all)
DIR_MODE = 0o777
FILE_MODE = 0o666

def _json_serializable(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_serializable(x) for x in obj]
    return obj

def _ensure_parent_dir(path: Path) -> None:
    """
    Ensure parent directory exists and set permissions.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, DIR_MODE)
    except Exception as e:
        print(f"⚠️  Could not chmod {parent} to {oct(DIR_MODE)}: {e}")

def _chmod_file(path: Path) -> None:
    """Set file permissions after writing."""
    try:
        os.chmod(path, FILE_MODE)
    except Exception as e:
        print(f"⚠️  Could not chmod {path} to {oct(FILE_MODE)}: {e}")

# ---------------------------------------------------------------------
# Paths API (as you already have)
from src.io.paths import (
    get_latest_artifacts_dir,
    get_model_path,
    get_feature_names_path,
    get_threshold_path,
    get_metrics_path,
)
# ---------------------------------------------------------------------

def save_model(model: Any, model_name: str = "model.pkl") -> Path:
    """Save trained model to artifacts/latest/<model_name>."""
    model_path = get_model_path(model_name)
    _ensure_parent_dir(model_path)
    print(f"Saving model to: {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    _chmod_file(model_path)
    return model_path

def save_feature_names(feature_names: List[str]) -> Path:
    """Save feature names to artifacts/latest/feature_names.json."""
    feature_path = get_feature_names_path()
    _ensure_parent_dir(feature_path)
    print(f"Saving feature names to: {feature_path}")
    with open(feature_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    _chmod_file(feature_path)
    return feature_path

def save_threshold(threshold: float) -> Path:
    """Save optimal threshold to artifacts/latest/threshold.json."""
    threshold_path = get_threshold_path()
    _ensure_parent_dir(threshold_path)
    print(f"Saving threshold to: {threshold_path}")
    # Use the key expected by run_serve.py
    data = {"optimal_threshold": float(threshold)}
    with open(threshold_path, "w") as f:
        json.dump(data, f, indent=2)
    _chmod_file(threshold_path)
    return threshold_path

def save_metrics(metrics: Dict[str, Any]) -> Path:
    """Save training metrics to artifacts/latest/metrics.json."""
    metrics_path = get_metrics_path()
    _ensure_parent_dir(metrics_path)
    print(f"Saving metrics to: {metrics_path}")
    serializable = _json_serializable(metrics)
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    _chmod_file(metrics_path)
    return metrics_path

def save_all_artifacts(
    model: Any,
    feature_names: List[str],
    threshold: float,
    metrics: Dict[str, Any],
    model_name: str = "model.pkl",
) -> Dict[str, Path]:
    """Save all required artifacts after training."""
    print("Saving training artifacts...")
    # Ensure artifacts base dir exists (and set perms)
    artifacts_dir = get_latest_artifacts_dir()
    try:
        os.chmod(Path(artifacts_dir), DIR_MODE)
    except Exception as e:
        print(f"⚠️  Could not chmod {artifacts_dir} to {oct(DIR_MODE)}: {e}")

    saved_paths = {
        "model": save_model(model, model_name),
        "feature_names": save_feature_names(feature_names),
        "threshold": save_threshold(threshold),
        "metrics": save_metrics(metrics),
    }
    print("All artifacts saved successfully!")
    return saved_paths

def load_model(model_name: str = "model.pkl") -> Any:
    """Load trained model from artifacts/latest/."""
    model_path = get_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def load_feature_names() -> List[str]:
    """Load feature names from artifacts/latest/feature_names.json."""
    feature_path = get_feature_names_path()
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature names not found at: {feature_path}")
    with open(feature_path, "r") as f:
        return json.load(f)

def load_threshold() -> float:
    """Load threshold from artifacts/latest/threshold.json."""
    threshold_path = get_threshold_path()
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold not found at: {threshold_path}")
    with open(threshold_path, "r") as f:
        data = json.load(f)
        # Backward-compat: accept either key
        return float(data.get("optimal_threshold", data.get("threshold")))

def load_metrics() -> Dict[str, Any]:
    """Load metrics from artifacts/latest/metrics.json."""
    metrics_path = get_metrics_path()
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found at: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)
