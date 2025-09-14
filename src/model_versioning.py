"""
Enhanced model versioning and registry system.

Provides comprehensive model versioning with metadata tracking,
performance history, and model registry integration.
"""

import json
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
try:
    import yaml
except ImportError:
    try:
        import pyyaml as yaml
    except ImportError:
        raise ImportError(
            "Neither 'yaml' nor 'pyyaml' is installed. Please install PyYAML:\n"
            "pip install pyyaml==6.0.1"
        )

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Comprehensive model metadata for versioning."""
    
    def __init__(self, model_name, model_type, version, created_at, training_date, training_duration, random_seed,
                 training_samples, validation_samples, test_samples, feature_count, target_column,
                 train_accuracy, val_accuracy, test_accuracy, train_f1, val_f1, test_f1,
                 train_auc, val_auc, test_auc, hyperparameters, feature_names, optimal_threshold,
                 preprocessing_steps, feature_selection_method, class_imbalance_method,
                 model_path, metadata_path, config_path, git_commit=None, environment_info=None, notes=None):
        # Basic model info
        self.model_name = model_name
        self.model_type = model_type
        self.version = version
        self.created_at = created_at
        
        # Training info
        self.training_date = training_date
        self.training_duration = training_duration
        self.random_seed = random_seed
        
        # Data info
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.test_samples = test_samples
        self.feature_count = feature_count
        self.target_column = target_column
        
        # Performance metrics
        self.train_accuracy = train_accuracy
        self.val_accuracy = val_accuracy
        self.test_accuracy = test_accuracy
        self.train_f1 = train_f1
        self.val_f1 = val_f1
        self.test_f1 = test_f1
        self.train_auc = train_auc
        self.val_auc = val_auc
        self.test_auc = test_auc
        
        # Model parameters
        self.hyperparameters = hyperparameters
        self.feature_names = feature_names
        self.optimal_threshold = optimal_threshold
        
        # Preprocessing info
        self.preprocessing_steps = preprocessing_steps
        self.feature_selection_method = feature_selection_method
        self.class_imbalance_method = class_imbalance_method
        
        # File paths
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.config_path = config_path
        
        # Additional metadata
        self.git_commit = git_commit
        self.environment_info = environment_info
        self.notes = notes
    
    def asdict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'created_at': self.created_at,
            'training_date': self.training_date,
            'training_duration': self.training_duration,
            'random_seed': self.random_seed,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'test_samples': self.test_samples,
            'feature_count': self.feature_count,
            'target_column': self.target_column,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'test_accuracy': self.test_accuracy,
            'train_f1': self.train_f1,
            'val_f1': self.val_f1,
            'test_f1': self.test_f1,
            'train_auc': self.train_auc,
            'val_auc': self.val_auc,
            'test_auc': self.test_auc,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'preprocessing_steps': self.preprocessing_steps,
            'feature_selection_method': self.feature_selection_method,
            'class_imbalance_method': self.class_imbalance_method,
            'model_path': self.model_path,
            'metadata_path': self.metadata_path,
            'config_path': self.config_path,
            'git_commit': self.git_commit,
            'environment_info': self.environment_info,
            'notes': self.notes
        }


class ModelVersioning:
    """Enhanced model versioning system with registry integration."""
    
    def __init__(self, registry_path="models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        (self.registry_path / "configs").mkdir(exist_ok=True)
        (self.registry_path / "artifacts").mkdir(exist_ok=True)
    
    def _generate_version_hash(self, model_name, timestamp, metadata):
        """Generate a unique version hash based on model name, timestamp, and key metadata."""
        # Create a hash from model name, timestamp, and key performance metrics
        hash_input = f"{model_name}_{timestamp}_{metadata.get('test_accuracy', 0):.4f}_{metadata.get('test_f1', 0):.4f}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def _get_git_commit(self):
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _get_environment_info(self):
        """Get current environment information."""
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor()
        }
    
    def save_model(self, 
                   model,
                   model_name,
                   model_type,
                   metadata,
                   config,
                   feature_names,
                   optimal_threshold,
                   preprocessing_steps,
                   notes=None):
        """
        Save a model with comprehensive versioning and metadata.
        
        Returns:
            str: Version hash of the saved model
        """
        timestamp = datetime.now().isoformat()
        version_hash = self._generate_version_hash(model_name, timestamp, metadata)
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version_hash,
            created_at=timestamp,
            training_date=timestamp,
            training_duration=metadata.get('training_duration', 0.0),
            random_seed=metadata.get('random_seed', 42),
            training_samples=metadata.get('training_samples', 0),
            validation_samples=metadata.get('validation_samples', 0),
            test_samples=metadata.get('test_samples', 0),
            feature_count=len(feature_names),
            target_column=metadata.get('target_column', 'alzheimers_diagnosis'),
            train_accuracy=metadata.get('train_accuracy', 0.0),
            val_accuracy=metadata.get('val_accuracy', 0.0),
            test_accuracy=metadata.get('test_accuracy', 0.0),
            train_f1=metadata.get('train_f1', 0.0),
            val_f1=metadata.get('val_f1', 0.0),
            test_f1=metadata.get('test_f1', 0.0),
            train_auc=metadata.get('train_auc', 0.0),
            val_auc=metadata.get('val_auc', 0.0),
            test_auc=metadata.get('test_auc', 0.0),
            hyperparameters=metadata.get('hyperparameters', {}),
            feature_names=feature_names,
            optimal_threshold=optimal_threshold,
            preprocessing_steps=preprocessing_steps,
            feature_selection_method=metadata.get('feature_selection_method', 'variance_threshold'),
            class_imbalance_method=metadata.get('class_imbalance_method', 'class_weight'),
            model_path=f"models/{model_name}_{version_hash}.pkl",
            metadata_path=f"metadata/{model_name}_{version_hash}.json",
            config_path=f"configs/{model_name}_{version_hash}.yaml",
            git_commit=self._get_git_commit(),
            environment_info=self._get_environment_info(),
            notes=notes
        )
        
        # Save model file
        model_path = self.registry_path / "models" / f"{model_name}_{version_hash}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = self.registry_path / "metadata" / f"{model_name}_{version_hash}.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata.asdict(), f, indent=2, default=str)
        
        # Save config
        config_path = self.registry_path / "configs" / f"{model_name}_{version_hash}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Update registry index
        self._update_registry_index(model_metadata)
        
        logger.info(f"Model {model_name} saved with version {version_hash}")
        return version_hash
    
    def _update_registry_index(self, metadata):
        """Update the registry index with new model information."""
        index_path = self.registry_path / "registry_index.json"
        
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"models": {}}
        
        # Add model to index
        if metadata.model_name not in index["models"]:
            index["models"][metadata.model_name] = []
        
        # Add version info
        version_info = {
            "version": metadata.version,
            "created_at": metadata.created_at,
            "test_accuracy": metadata.test_accuracy,
            "test_f1": metadata.test_f1,
            "test_auc": metadata.test_auc,
            "model_path": metadata.model_path,
            "metadata_path": metadata.metadata_path,
            "config_path": metadata.config_path
        }
        
        index["models"][metadata.model_name].append(version_info)
        
        # Sort versions by creation date (newest first)
        index["models"][metadata.model_name].sort(
            key=lambda x: x["created_at"], reverse=True
        )
        
        # Keep only last 10 versions per model
        index["models"][metadata.model_name] = index["models"][metadata.model_name][:10]
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def load_model(self, model_name, version=None):
        """
        Load a model and its metadata.
        
        Args:
            model_name: Name of the model
            version: Specific version to load (if None, loads latest)
            
        Returns:
            tuple: (model, metadata, config)
        """
        index_path = self.registry_path / "registry_index.json"
        
        if not index_path.exists():
            raise FileNotFoundError("Model registry not found")
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        if model_name not in index["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        versions = index["models"][model_name]
        
        if version is None:
            # Load latest version
            version_info = versions[0]
        else:
            # Load specific version
            version_info = next((v for v in versions if v["version"] == version), None)
            if version_info is None:
                raise ValueError(f"Version {version} not found for model {model_name}")
        
        # Load model
        model_path = self.registry_path / version_info["model_path"]
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_path = self.registry_path / version_info["metadata_path"]
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load config
        config_path = self.registry_path / version_info["config_path"]
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded model {model_name} version {version_info['version']}")
        return model, metadata, config
    
    def list_models(self):
        """List all models and their versions in the registry."""
        index_path = self.registry_path / "registry_index.json"
        
        if not index_path.exists():
            return {}
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        return index["models"]
    
    def get_model_history(self, model_name):
        """Get training history for a specific model."""
        models = self.list_models()
        
        if model_name not in models:
            return []
        
        return models[model_name]
    
    def compare_models(self, model_name, version1, version2):
        """Compare two versions of the same model."""
        history = self.get_model_history(model_name)
        
        v1_info = next((v for v in history if v["version"] == version1), None)
        v2_info = next((v for v in history if v["version"] == version2), None)
        
        if not v1_info or not v2_info:
            raise ValueError("One or both versions not found")
        
        comparison = {
            "model_name": model_name,
            "version1": {
                "version": v1_info["version"],
                "created_at": v1_info["created_at"],
                "test_accuracy": v1_info["test_accuracy"],
                "test_f1": v1_info["test_f1"],
                "test_auc": v1_info["test_auc"]
            },
            "version2": {
                "version": v2_info["version"],
                "created_at": v2_info["created_at"],
                "test_accuracy": v2_info["test_accuracy"],
                "test_f1": v2_info["test_f1"],
                "test_auc": v2_info["test_auc"]
            },
            "improvements": {
                "accuracy_delta": v2_info["test_accuracy"] - v1_info["test_accuracy"],
                "f1_delta": v2_info["test_f1"] - v1_info["test_f1"],
                "auc_delta": v2_info["test_auc"] - v1_info["test_auc"]
            }
        }
        
        return comparison
