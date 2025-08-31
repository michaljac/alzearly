"""
Configuration management for the ML project.
Loads and validates configuration from YAML files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

# YAML import with fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    try:
        import pyyaml as yaml
        YAML_AVAILABLE = True
    except ImportError:
        YAML_AVAILABLE = False
        print("⚠️  Warning: Neither 'yaml' nor 'pyyaml' is installed.")
        print("   Using default configuration values.")
        print("   To install: pip install pyyaml==6.0.1")

logger = logging.getLogger(__name__)


@dataclass
class DataGenConfig:
    """Configuration for data generation."""
    n_patients: int
    years: list[int]
    positive_rate: float = 0.08
    rows_per_chunk: int = 100_000
    seed: int = 42
    output_dir: str = "/Data/raw"
    target_column: str = "alzheimers_diagnosis"
    clinical_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    risk_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    input_dir: str = "/Data/raw"
    output_dir: str = "/Data/featurized"
    rolling_window_years: int = 3
    numeric_columns: Optional[list[str]] = None
    categorical_columns: Optional[list[str]] = None
    binary_columns: Optional[list[str]] = None
    chunk_size: int = 100_000
    seed: int = 42
    target_column: str = "alzheimers_diagnosis"
    risk_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    categorical_encoding: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # Data configuration
    input_dir: str = "/Data/featurized"
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
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = False
    n_trials: int = 50
    
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


class ConfigLoader:
    """Loads and validates configuration from YAML files."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory {config_dir} does not exist")
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        # Handle full paths vs relative paths
        if "/" in filename:
            filepath = Path(filename)
        else:
            filepath = self.config_dir / filename
            
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file {filepath} does not exist")
        
        if not YAML_AVAILABLE:
            # Return default configuration when YAML is not available
            print(f"⚠️  YAML not available, using default configuration for {filename}")
            return self._get_default_config(filename)
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Loaded configuration from file
        return config
    
    def _get_default_config(self, filename: str) -> Dict[str, Any]:
        """Get default configuration when YAML is not available."""
        if "data_gen" in filename:
            return {
                "dataset": {
                    "n_patients": 3000,
                    "years": [2021, 2022, 2023, 2024],
                    "target_rows": 12000
                },
                "target": {
                    "positive_rate": 0.08,
                    "column_name": "alzheimers_diagnosis"
                },
                "processing": {
                    "rows_per_chunk": 100000,
                    "seed": 42
                },
                "output": {
                    "directory": "/Data/raw"
                },
                "clinical_ranges": {},
                "risk_factors": {}
            }
        elif "model" in filename:
            return {
                "input_dir": "/Data/featurized",
                "target_column": "alzheimers_diagnosis",
                "exclude_columns": ["patient_id", "year"],
                "test_size": 0.2,
                "val_size": 0.2,
                "random_state": 42,
                "stratify": True,
                "max_features": 150,
                "variance_threshold": 0.01,
                "handle_imbalance": "class_weight",
                "models": ["logistic_regression", "xgboost"],
                "enable_hyperparameter_tuning": False,
                "n_trials": 50,
                "xgb_params": {},
                "lr_params": {"random_state": 42, "max_iter": 1000},
                "output_dir": "models",
                "save_metadata": True,
                "wandb_project": "alzheimers-prediction",
                "wandb_entity": None,
                "log_artifacts": True
            }
        else:
            return {}
    
    def load_data_gen_config(self, filename: str = "data_gen.yaml") -> DataGenConfig:
        """Load data generation configuration."""
        config = self.load_yaml(filename)
        
        # Extract dataset configuration
        dataset = config.get("dataset", {})
        n_patients = dataset.get("n_patients", 100000)
        years = dataset.get("years", [2018, 2019, 2020])
        target_rows = dataset.get("target_rows")
        
        # Override n_patients if target_rows is specified
        if target_rows:
            n_patients = target_rows // len(years)
            # Calculated n_patients from target_rows
        
        # Extract target configuration
        target = config.get("target", {})
        positive_rate = target.get("positive_rate", 0.08)
        target_column = target.get("column_name", "alzheimers_diagnosis")
        
        # Extract processing configuration
        processing = config.get("processing", {})
        rows_per_chunk = processing.get("rows_per_chunk", 100000)
        seed = processing.get("seed", 42)
        
        # Extract output configuration
        output = config.get("output", {})
        output_dir = output.get("directory", "/Data/raw")
        
        # Extract clinical ranges and risk factors
        clinical_ranges = config.get("clinical_ranges", {})
        risk_factors = config.get("risk_factors", {})
        
        return DataGenConfig(
            n_patients=n_patients,
            years=years,
            positive_rate=positive_rate,
            rows_per_chunk=rows_per_chunk,
            seed=seed,
            output_dir=output_dir,
            target_column=target_column,
            clinical_ranges=clinical_ranges,
            risk_factors=risk_factors
        )
    
    def load_preprocess_config(self, filename: str = "preprocess.yaml") -> PreprocessConfig:
        """Load preprocessing configuration."""
        config = self.load_yaml(filename)
        
        # Extract IO configuration
        io = config.get("io", {})
        input_dir = io.get("input_dir", "/Data/raw")
        output_dir = io.get("output_dir", "/Data/featurized")
        
        # Extract feature configuration
        features = config.get("features", {})
        target_column = features.get("target_column", "alzheimers_diagnosis")
        numeric_columns = features.get("numeric_columns")
        categorical_columns = features.get("categorical_columns")
        binary_columns = features.get("binary_columns")
        
        # Extract rolling configuration
        rolling = config.get("rolling", {})
        rolling_window_years = rolling.get("window_years", 3)
        
        # Extract processing configuration
        processing = config.get("processing", {})
        chunk_size = processing.get("chunk_size", 100000)
        seed = processing.get("seed", 42)
        
        # Extract risk feature thresholds
        risk_features = config.get("risk_features", {})
        
        # Extract categorical encoding configuration
        categorical_encoding = config.get("categorical_encoding", {})
        
        return PreprocessConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            rolling_window_years=rolling_window_years,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            chunk_size=chunk_size,
            seed=seed,
            target_column=target_column,
            risk_thresholds=risk_features,
            categorical_encoding=categorical_encoding
        )
    
    def load_model_config(self, filename: str = "model.yaml") -> ModelConfig:
        """Load model training configuration."""
        config = self.load_yaml(filename)
        
        # Load flat structure (new format)
        input_dir = config.get("input_dir", "/Data/featurized")
        target_column = config.get("target_column", "alzheimers_diagnosis")
        exclude_columns = config.get("exclude_columns", ["patient_id", "year"])
        
        test_size = config.get("test_size", 0.2)
        val_size = config.get("val_size", 0.2)
        random_state = config.get("random_state", 42)
        stratify = config.get("stratify", True)
        
        max_features = config.get("max_features", 150)
        variance_threshold = config.get("variance_threshold", 0.01)
        
        handle_imbalance = config.get("handle_imbalance", "class_weight")
        
        # Handle both flat and nested models configuration
        models_config = config.get("models", {})
        if isinstance(models_config, list):
            models = models_config
        else:
            models = models_config.get("types", ["logistic_regression", "xgboost"])
        
        # Hyperparameter tuning
        enable_hyperparameter_tuning = config.get("enable_hyperparameter_tuning", False)
        n_trials = config.get("n_trials", 50)
        
        # XGBoost parameters
        xgb_params = config.get("xgb_params", {})
        
        # Logistic Regression parameters
        lr_params = config.get("lr_params", {
            "random_state": random_state,
            "max_iter": 1000
        })
        
        output_dir = config.get("output_dir", "models")
        save_metadata = config.get("save_metadata", True)
        
        wandb_project = config.get("wandb_project", "alzheimers-prediction")
        wandb_entity = config.get("wandb_entity", None)
        log_artifacts = config.get("log_artifacts", True)
        
        return ModelConfig(
            input_dir=input_dir,
            target_column=target_column,
            exclude_columns=exclude_columns,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stratify=stratify,
            max_features=max_features,
            variance_threshold=variance_threshold,
            handle_imbalance=handle_imbalance,
            models=models,
            enable_hyperparameter_tuning=enable_hyperparameter_tuning,
            n_trials=n_trials,
            xgb_params=xgb_params,
            lr_params=lr_params,
            output_dir=output_dir,
            save_metadata=save_metadata,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            log_artifacts=log_artifacts
        )


def load_config(config_type: str, filename: Optional[str] = None) -> Any:
    """Convenience function to load configuration by type."""
    loader = ConfigLoader()
    
    if config_type == "data_gen":
        return loader.load_data_gen_config(filename or "data_gen.yaml")
    elif config_type == "preprocess":
        return loader.load_preprocess_config(filename or "preprocess.yaml")
    elif config_type == "model":
        return loader.load_model_config(filename or "model.yaml")
    else:
        raise ValueError(f"Unknown config type: {config_type}")
