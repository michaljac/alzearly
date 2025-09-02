"""
Configuration management for the ML project.
Loads and validates configuration from YAML files.
"""

import yaml
from pathlib import Path


class DataGenConfig:
    """Configuration for data generation."""
    n_patients = int
    years = list
    positive_rate = 0.08
    rows_per_chunk = 100_000
    seed = 42
    output_dir = "/Data/raw"
    target_column = "alzheimers_diagnosis"
    clinical_ranges = dict
    risk_factors = dict


class PreprocessConfig:
    """Configuration for data preprocessing."""
    input_dir = "/Data/raw"
    output_dir = "/Data/featurized"
    rolling_window_years = 3
    numeric_columns = list
    categorical_columns = list
    binary_columns = list
    chunk_size = 100_000
    seed = 42
    target_column = "alzheimers_diagnosis"
    risk_thresholds = dict
    categorical_encoding = dict


class ModelConfig:
    """Configuration for model training."""
    input_dir = "/Data/featurized"
    target_column = "alzheimers_diagnosis"
    exclude_columns = ["patient_id", "year"]

    test_size = 0.2
    val_size = 0.2
    random_state = 42
    stratify = True

    max_features = 150
    variance_threshold = 0.01

    handle_imbalance = "class_weight"
    models = ["logistic_regression", "xgboost"]

    enable_hyperparameter_tuning = False
    n_trials = 50

    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    lr_params = {
        "random_state": 42,
        "max_iter": 1000,
    }

    output_dir = "models"
    save_metadata = True
    log_artifacts = True


class ConfigLoader:
    """Loads YAML configuration files into dicts."""

    def __init__(self, config_dir="config"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory {config_dir} does not exist")

    def load_yaml(self, filename: str):
        filepath = Path(filename) if "/" in filename else self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file {filepath} does not exist")
        with open(filepath, "r") as f:
            return yaml.safe_load(f)

    def load_data_gen_config(self, filename="data_gen.yaml") -> dict:
        return self.load_yaml(filename)

    def load_preprocess_config(self, filename="preprocess.yaml") -> dict:
        return self.load_yaml(filename)

    def load_model_config(self, filename="model.yaml") -> dict:
        return self.load_yaml(filename)


def load_config(config_type: str, filename=None):
    loader = ConfigLoader()
    if config_type == "data_gen":
        return loader.load_data_gen_config(filename or "data_gen.yaml")
    elif config_type == "preprocess":
        return loader.load_preprocess_config(filename or "preprocess.yaml")
    elif config_type == "model":
        return loader.load_model_config(filename or "model.yaml")
    else:
        raise ValueError(f"Unknown config type: {config_type}")
