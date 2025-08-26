"""
Path management utilities for the Alzearly pipeline.
Provides consistent paths for artifacts, models, data, and other resources.
"""

import os
from pathlib import Path


# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"

# Artifacts and models
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LATEST_ARTIFACTS_DIR = ARTIFACTS_DIR / "latest"
MODELS_DIR = PROJECT_ROOT / "models"

# Experiment tracking
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
WANDB_DIR = PROJECT_ROOT / "wandb"

# Visualization
PLOTS_DIR = PROJECT_ROOT / "plots"
README_IMAGES_DIR = PROJECT_ROOT / "readme_images"

# Configuration
CONFIG_DIR = PROJECT_ROOT / "config"

# Source code
SRC_DIR = PROJECT_ROOT / "src"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_artifacts_dir() -> Path:
    """Get the latest artifacts directory, create if it doesn't exist."""
    return ensure_dir(LATEST_ARTIFACTS_DIR)


def get_model_path(model_name: str = "model.pkl") -> Path:
    """Get path for model file in latest artifacts."""
    return get_latest_artifacts_dir() / model_name


def get_feature_names_path() -> Path:
    """Get path for feature names JSON file."""
    return get_latest_artifacts_dir() / "feature_names.json"


def get_threshold_path() -> Path:
    """Get path for threshold JSON file."""
    return get_latest_artifacts_dir() / "threshold.json"


def get_metrics_path() -> Path:
    """Get path for metrics JSON file."""
    return get_latest_artifacts_dir() / "metrics.json"


def get_plot_path(plot_name: str) -> Path:
    """Get path for plot file."""
    return ensure_dir(PLOTS_DIR) / plot_name


def get_config_path(config_name: str) -> Path:
    """Get path for configuration file."""
    return CONFIG_DIR / config_name


# Ensure essential directories exist
ensure_dir(DATA_DIR)
ensure_dir(ARTIFACTS_DIR)
ensure_dir(PLOTS_DIR)
ensure_dir(CONFIG_DIR)
