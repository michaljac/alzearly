"""
Configuration management for the ML project.
Loads and validates configuration from YAML files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DataGenConfig:
    """Configuration for data generation."""
    n_patients: int
    years: list[int]
    positive_rate: float = 0.08
    rows_per_chunk: int = 100_000
    seed: int = 42
    output_dir: str = "data/raw"
    target_column: str = "alzheimers_diagnosis"
    clinical_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    risk_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""
    input_dir: str = "data/raw"
    output_dir: str = "data/featurized"
    rolling_window_years: int = 3
    numeric_columns: Optional[list[str]] = None
    chunk_size: int = 100_000
    seed: int = 42
    target_column: str = "alzheimers_diagnosis"
    risk_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = "xgboost"
    model_name: str = "alzheimers_predictor"
    test_size: float = 0.2
    validation_size: float = 0.2
    seed: int = 42
    cv_folds: int = 5
    stratified: bool = True
    feature_selection_enabled: bool = True
    feature_selection_method: str = "mutual_info"
    n_features: int = 50
    hyperparameter_tuning_enabled: bool = True
    hyperparameter_tuning_method: str = "bayesian"
    n_trials: int = 100
    primary_metric: str = "roc_auc"
    input_dir: str = "data/featurized"
    target_column: str = "alzheimers_diagnosis"
    model_dir: str = "models"
    results_dir: str = "results"


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
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {filepath}")
        return config
    
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
            logger.info(f"Calculated n_patients={n_patients} from target_rows={target_rows}")
        
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
        output_dir = output.get("directory", "data/raw")
        
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
        input_dir = io.get("input_dir", "data/raw")
        output_dir = io.get("output_dir", "data/featurized")
        
        # Extract feature configuration
        features = config.get("features", {})
        target_column = features.get("target_column", "alzheimers_diagnosis")
        numeric_columns = features.get("numeric_columns")
        
        # Extract rolling configuration
        rolling = config.get("rolling", {})
        rolling_window_years = rolling.get("window_years", 3)
        
        # Extract processing configuration
        processing = config.get("processing", {})
        chunk_size = processing.get("chunk_size", 100000)
        seed = processing.get("seed", 42)
        
        # Extract risk feature thresholds
        risk_features = config.get("risk_features", {})
        
        return PreprocessConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            rolling_window_years=rolling_window_years,
            numeric_columns=numeric_columns,
            chunk_size=chunk_size,
            seed=seed,
            target_column=target_column,
            risk_thresholds=risk_features
        )
    
    def load_model_config(self, filename: str = "model.yaml") -> ModelConfig:
        """Load model training configuration."""
        config = self.load_yaml(filename)
        
        # Extract model configuration
        model = config.get("model", {})
        model_type = model.get("type", "xgboost")
        model_name = model.get("name", "alzheimers_predictor")
        
        # Extract training configuration
        training = config.get("training", {})
        test_size = training.get("test_size", 0.2)
        validation_size = training.get("validation_size", 0.2)
        seed = training.get("seed", 42)
        cv_folds = training.get("cv_folds", 5)
        stratified = training.get("stratified", True)
        
        # Extract feature selection configuration
        feature_selection = config.get("feature_selection", {})
        feature_selection_enabled = feature_selection.get("enabled", True)
        feature_selection_method = feature_selection.get("method", "mutual_info")
        n_features = feature_selection.get("n_features", 50)
        
        # Extract hyperparameter tuning configuration
        hyperparameter_tuning = config.get("hyperparameter_tuning", {})
        hyperparameter_tuning_enabled = hyperparameter_tuning.get("enabled", True)
        hyperparameter_tuning_method = hyperparameter_tuning.get("method", "bayesian")
        n_trials = hyperparameter_tuning.get("n_trials", 100)
        
        # Extract evaluation configuration
        evaluation = config.get("evaluation", {})
        primary_metric = evaluation.get("primary_metric", "roc_auc")
        
        # Extract data configuration
        data = config.get("data", {})
        input_dir = data.get("input_dir", "data/featurized")
        target_column = data.get("target_column", "alzheimers_diagnosis")
        
        # Extract output configuration
        output = config.get("output", {})
        model_dir = output.get("model_dir", "models")
        results_dir = output.get("results_dir", "results")
        
        return ModelConfig(
            model_type=model_type,
            model_name=model_name,
            test_size=test_size,
            validation_size=validation_size,
            seed=seed,
            cv_folds=cv_folds,
            stratified=stratified,
            feature_selection_enabled=feature_selection_enabled,
            feature_selection_method=feature_selection_method,
            n_features=n_features,
            hyperparameter_tuning_enabled=hyperparameter_tuning_enabled,
            hyperparameter_tuning_method=hyperparameter_tuning_method,
            n_trials=n_trials,
            primary_metric=primary_metric,
            input_dir=input_dir,
            target_column=target_column,
            model_dir=model_dir,
            results_dir=results_dir
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
