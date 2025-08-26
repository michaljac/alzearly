#!/usr/bin/env python3
"""
Main script for Alzheimer's Prediction Pipeline.

This script provides an interactive interface to run the complete ML pipeline
from data generation to model serving.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import load_config
from src.data_gen import SyntheticDataGenerator
from src.preprocess import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from utils import (
    get_input, setup_logging, check_dependencies, create_directories,
    handle_critical_error, handle_recoverable_error, setup_experiment_tracker
)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


def run_data_generation(config_file: str = "config/data_gen.yaml", auto_confirm: bool = False) -> bool:
    """Run data generation step."""
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)
    
    try:
        # Load configuration
        config = load_config("data_gen", config_file)
        
        # Get user confirmation (skip if auto_confirm is True)
        if not auto_confirm:
            if get_input("Proceed with data generation? (y/n)", input_type="y/n") != 'y':
                return False
        
        # Create directories
        create_directories([config.output_dir])
        
        # Generate data
        generator = SyntheticDataGenerator(config)
        generator.generate()
        
        print("✅ Data generation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        print(f"❌ Data generation failed: {e}")
        return False


def run_preprocessing(config_file: str = "config/preprocess.yaml", auto_confirm: bool = False) -> bool:
    """Run data preprocessing step."""
    print("\n" + "="*60)
    print("STEP 2: DATA PREPROCESSING")
    print("="*60)
    
    try:
        # Check if raw data exists
        raw_data_path = Path("data/raw")
        if not raw_data_path.exists():
            print("❌ Raw data not found. Please run data generation first (Step 1).")
            print("   Expected location: data/raw/")
            return False
        
        # Load configuration
        config = load_config("preprocess", config_file)
        
        # Get user confirmation (skip if auto_confirm is True)
        if not auto_confirm:
            if get_input("Proceed with data preprocessing?", input_type="y/n") != 'y':
                return False
        
        # Create directories
        create_directories([config.output_dir])
        
        # Preprocess data
        preprocessor = DataPreprocessor(config)
        preprocessor.preprocess()
        
        print("✅ Data preprocessing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        print(f"❌ Data preprocessing failed: {e}")
        return False


def run_training(config_file: str = "config/model.yaml", auto_confirm: bool = False) -> Optional[str]:
    """Run model training step."""
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    try:
        # Check if featurized data exists
        featurized_data_path = Path("data/featurized")
        if not featurized_data_path.exists():
            print("❌ Featurized data not found. Please run data preprocessing first (Step 2).")
            print("   Expected location: data/featurized/")
            return None
        
        # Load configuration
        config = load_config("model", config_file)
        
        # Get user confirmation (skip if auto_confirm is True)
        if not auto_confirm:
            if get_input("Proceed with model training?", input_type="y/n") != 'y':
                return None
        
        # Create directories
        create_directories([config.output_dir])
        
        # Train models
        trainer = ModelTrainer(config)
        results = trainer.train("initial")  # Use "initial" as default run type
        
        print("✅ Model training completed successfully!")
        
        return results['artifact_path']
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        print(f"❌ Model training failed: {e}")
        return None


def run_evaluation(model_path: str, data_path: str, output_dir: str = "artifacts") -> bool:
    """Run model evaluation step."""
    print("\n" + "="*60)
    print("STEP 4: MODEL EVALUATION")
    print("="*60)
    
    try:
        # Check if models exist
        models_path = Path("models")
        if not models_path.exists():
            print("❌ No trained models found. Please run model training first (Step 3).")
            print("   Expected location: models/")
            return False
        
        # Check if featurized data exists
        featurized_data_path = Path("data/featurized")
        if not featurized_data_path.exists():
            print("❌ Featurized data not found. Please run data preprocessing first (Step 2).")
            print("   Expected location: data/featurized/")
            return False
        
        # Get user confirmation
        if get_input("Proceed with model evaluation?", input_type="y/n") != 'y':
            return False
        
        # Create directories
        create_directories([output_dir])
        
        # Handle model path - if it's a directory, find the first model file
        model_file_path = Path(model_path)
        if model_file_path.is_dir():
            # Look for model files in the directory
            model_files = list(model_file_path.glob("*.pkl"))
            if model_files:
                model_file_path = model_files[0]  # Use the first model file found
                print(f"📁 Found model file: {model_file_path}")
            else:
                raise FileNotFoundError(f"No .pkl model files found in {model_path}")
        elif not model_file_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Evaluate model
        evaluator = ModelEvaluator(str(model_file_path), data_path, output_dir)
        results = evaluator.evaluate()
        
        print("✅ Model evaluation completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        print(f"❌ Model evaluation failed: {e}")
        return False


def run_serve_dev() -> bool:
    """Run development server."""
    print("\n" + "="*60)
    print("STEP 5: DEVELOPMENT SERVER")
    print("="*60)
    
    try:
        # Check if models exist
        models_path = Path("models")
        if not models_path.exists():
            print("❌ No trained models found. Please run model training first (Step 3).")
            print("   Expected location: models/")
            return False
        
        # Check if FastAPI dependencies are available
        try:
            import fastapi
            import uvicorn
        except ImportError:
            print("❌ FastAPI dependencies not available in training environment")
            print("   To run the development server, use Docker Compose:")
            print("   docker-compose --profile serve up")
            print("   ")
            print("   Or run the serve service directly:")
            print("   docker-compose run --service-ports serve")
            print("   ")
            print("   The API will be available at: http://localhost:8000")
            print("   API documentation at: http://localhost:8000/docs")
            return False
        
        # Get user confirmation
        if get_input("Start development server?", input_type="y/n") != 'y':
            return False
        
        # Import and run the serve module
        from src.serve import app
        
        print("🚀 Starting FastAPI server...")
        
        uvicorn.run(
            "src.serve:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"Development server failed: {e}")
        print(f"❌ Development server failed: {e}")
        return False


def run_comparison_plots() -> bool:
    """Create comparison plots for XGBoost and Logistic Regression."""
    print("\n" + "="*60)
    print("STEP 6: CREATE COMPARISON PLOTS")
    print("="*60)
    
    try:
        # Check if models exist
        models_path = Path("models")
        if not models_path.exists():
            print("❌ No trained models found. Please run model training first (Step 3).")
            print("   Expected location: models/")
            return False
        
        # Check if featurized data exists
        featurized_data_path = Path("data/featurized")
        if not featurized_data_path.exists():
            print("❌ Featurized data not found. Please run data preprocessing first (Step 2).")
            print("   Expected location: data/featurized/")
            return False
        
        # Get user confirmation
        if get_input("Create model comparison plots?", input_type="y/n") != 'y':
            return False
        
        print("📊 Creating comparison plots...")
        
        # Set up the figure
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Performance Comparison', fontsize=20, fontweight='bold', y=0.95)
        
        # Find the latest training results
        plots_dir = Path("plots")
        if not plots_dir.exists():
            print("❌ No plots directory found. Please run training first.")
            return False
        
        # Find the most recent training run
        training_dirs = [d for d in plots_dir.iterdir() if d.is_dir()]
        if not training_dirs:
            print("❌ No training results found. Please run training first.")
            return False
        
        latest_dir = max(training_dirs, key=lambda x: len(list(x.glob("*.jpg"))))
        print(f"📁 Using training results from: {latest_dir.name}")
        
        # Try to create actual comparison plots
        try:
            models_dir = Path("models")
            data_dir = Path("data/featurized")
            if models_dir.exists() and data_dir.exists():
                create_actual_comparison_plots(ax1, ax2, data_dir, models_dir)
            else:
                load_existing_plots(ax1, ax2, latest_dir)
        except Exception as e:
            print(f"⚠️  Warning: Could not create actual plots: {e}")
            load_existing_plots(ax1, ax2, latest_dir)
        
        # Save the comparison plot with meaningful name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = plots_dir / f"model_comparison_{timestamp}.jpg"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Comparison plot saved to: {output_path}")
        
        plt.show()
        return True
        
    except Exception as e:
        logger.error(f"Comparison plots failed: {e}")
        print(f"❌ Comparison plots failed: {e}")
        return False


def create_actual_comparison_plots(ax1, ax2, data_dir, models_dir):
    """Create actual ROC and PR curves by loading models and data."""
    print("📊 Loading models and data for actual comparison...")
    
    # Load test data
    test_files = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            test_files.extend(list(subdir.glob("*.parquet")))
    
    if not test_files:
        print("❌ No test data found")
        return
    
    test_data = pd.read_parquet(test_files[0])
    print(f"📂 Loaded test data: {test_data.shape}")
    
    # Load models and metadata
    models = {}
    metadata = None
    
    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            metadata_file = subdir / "metadata.json"
            if metadata_file.exists() and metadata is None:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"✅ Loaded metadata from {subdir.name}")
            
            model_files = list(subdir.glob("*.pkl"))
            for model_file in model_files:
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    model_name = model_file.stem
                    models[model_name] = model
                    print(f"✅ Loaded model: {model_name}")
                except Exception as e:
                    print(f"⚠️  Could not load {model_file}: {e}")
    
    if not models:
        print("❌ No models found")
        return
    
    # Prepare features using metadata
    if metadata and 'feature_names' in metadata:
        feature_names = metadata['feature_names']
        print(f"📊 Using {len(feature_names)} features from metadata")
        
        available_features = [col for col in feature_names if col in test_data.columns]
        missing_features = [col for col in feature_names if col not in test_data.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {len(missing_features)}")
            for feature in missing_features:
                test_data[feature] = 0
        
        X_test = test_data[feature_names].fillna(0).values
        y_test = test_data['alzheimers_diagnosis'].values
        print(f"📊 Prepared features: {X_test.shape}")
    else:
        print("❌ No metadata found, using fallback feature preparation")
        exclude_cols = ['patient_id', 'year', 'alzheimers_diagnosis']
        feature_cols = [col for col in test_data.columns if col not in exclude_cols]
        X_test = test_data[feature_cols].fillna(0).values
        y_test = test_data['alzheimers_diagnosis'].values
    
    # Generate predictions and create curves
    colors = {'xgboost': 'orange', 'logistic_regression': 'blue'}
    
    # ROC Curve
    ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    for model_name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            color = colors.get(model_name, 'green')
            label = f"{model_name.replace('_', ' ').title()} (AUC={auc_score:.2f})"
            ax1.plot(fpr, tpr, color=color, lw=2, label=label)
        except Exception as e:
            print(f"⚠️  Error with {model_name}: {e}")
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    ax2.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    
    for model_name, model in tqdm(models.items(), 
                                 desc="Generating comparison plots", 
                                 unit="model",
                                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                                 ncols=80, ascii=True, position=0, dynamic_ncols=False, 
                                 mininterval=0.1, maxinterval=1.0):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            
            color = colors.get(model_name, 'green')
            label = f"{model_name.replace('_', ' ').title()} (AP={ap_score:.2f})"
            ax2.plot(recall, precision, color=color, lw=2, label=label)
        except Exception as e:
            print(f"⚠️  Error with {model_name}: {e}")
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)


def load_existing_plots(ax1, ax2, latest_dir):
    """Load existing plot images as fallback."""
    print("📊 Loading existing plot images...")
    
    try:
        xgb_roc = mpimg.imread(latest_dir / "xgboost_roc_curve.jpg")
        ax1.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
        ax1.imshow(xgb_roc)
        ax1.axis('off')
    except FileNotFoundError as e:
        print(f"⚠️  Warning: Could not load ROC curves: {e}")
        ax1.text(0.5, 0.5, 'ROC Curves\n(Not available)', 
                ha='center', va='center', transform=ax1.transAxes)
    
    try:
        xgb_pr = mpimg.imread(latest_dir / "xgboost_pr_curve.jpg")
        ax2.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
        ax2.imshow(xgb_pr)
        ax2.axis('off')
    except FileNotFoundError as e:
        print(f"⚠️  Warning: Could not load PR curves: {e}")
        ax2.text(0.5, 0.5, 'PR Curves\n(Not available)', 
                ha='center', va='center', transform=ax2.transAxes)


def run_tests() -> bool:
    """Run project tests."""
    print("\n" + "="*60)
    print("STEP 7: RUN TESTS")
    print("="*60)
    
    try:
        # Get user confirmation
        if get_input("Run project tests?", input_type="y/n") != 'y':
            return False
        
        print("🧪 Running project tests...")
        
        # Test imports
        print("📦 Testing imports...")
        try:
            from src.config import load_config
            from src.data_gen import SyntheticDataGenerator
            from src.preprocess import DataPreprocessor
            from src.train import ModelTrainer
            from src.evaluate import ModelEvaluator
            print("✅ All imports successful")
        except Exception as e:
            print(f"❌ Import test failed: {e}")
            return False
        
        # Test configuration loading
        print("⚙️  Testing configuration loading...")
        try:
            config = load_config("data_gen", "config/data_gen.yaml")
            print("✅ Configuration loading successful")
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            return False
        
        # Test directory creation
        print("📁 Testing directory creation...")
        try:
            test_dir = Path("test_temp")
            test_dir.mkdir(exist_ok=True)
            test_dir.rmdir()
            print("✅ Directory operations successful")
        except Exception as e:
            print(f"❌ Directory test failed: {e}")
            return False
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        print(f"❌ Tests failed: {e}")
        return False


def main():
    """Main pipeline execution."""
    print("🧠 Alzheimer's Prediction Pipeline")
    print("="*60)
    
    # Setup logging
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        handle_critical_error(Exception("Dependency check failed"), "dependency validation")
    
    print("✅ All dependencies are available")
    
    # Setup experiment tracking
    from utils import setup_experiment_tracker
    global_tracker, global_tracker_type = setup_experiment_tracker()
    
    # Update global variables in utils module
    import utils
    utils.tracker = global_tracker
    utils.tracker_type = global_tracker_type
    
    # Pipeline steps
    steps = [
        ("Data Generation", run_data_generation),
        ("Data Preprocessing", run_preprocessing),
        ("Model Training", run_training),
        ("Model Evaluation", lambda: run_evaluation("models", "data/featurized")),
        ("Development Server", run_serve_dev),
        ("Create Comparison Plots", run_comparison_plots),
        ("Run Tests", run_tests)
    ]
    
    # Ask user which steps to run with robust input handling
    while True:
        print("\nAvailable pipeline steps:")
        for i, (name, _) in enumerate(steps, 1):
            print(f"{i}. {name}")
        print("0. Run all steps")
        
        choice = get_input("Enter step number (0-7) (or 'q' to quit)", allow_quit=True, input_type="choice", valid_choices=['0', '1', '2', '3', '4', '5', '6', '7'])
        
        if choice == 'q':
            print("👋 Goodbye!")
            return
        
        try:
            choice = int(choice)
            if choice == 0:
                # Run all steps
                print("\n🔄 Running complete pipeline...")
                print("🚀 Starting automated pipeline execution...")
                
                # Step 1: Data Generation (auto-confirm)
                print("\n📊 Step 1/5: Data Generation")
                if not run_data_generation(auto_confirm=True):
                    print("❌ Data generation failed. Pipeline stopped.")
                    return
                
                # Step 2: Preprocessing (auto-confirm)
                print("\n🔧 Step 2/5: Data Preprocessing")
                if not run_preprocessing(auto_confirm=True):
                    print("❌ Preprocessing failed. Pipeline stopped.")
                    return
                
                # Step 3: Training (auto-confirm)
                print("\n🤖 Step 3/5: Model Training")
                model_path = run_training(auto_confirm=True)
                if not model_path:
                    print("❌ Training failed. Pipeline stopped.")
                    return
                
                # Step 4: Evaluation (auto-confirm)
                print("\n📈 Step 4/5: Model Evaluation")
                if not run_evaluation(model_path, "data/featurized"):
                    print("❌ Evaluation failed. Pipeline stopped.")
                    return
                
                # Step 5: Development Server (skip in training container)
                print("\n🌐 Step 5/5: Development Server")
                print("ℹ️  Skipping development server in training container.")
                print("   To run the API server, use: docker-compose --profile serve up")
                
                print("\n🎉 Complete pipeline executed successfully!")
                print("📊 Next: Start the API server with: docker-compose --profile serve up")
                break  # Exit the loop after successful completion
                
            elif 1 <= choice <= len(steps):
                # Run specific step
                step_name, step_func = steps[choice - 1]
                print(f"\n🔄 Running: {step_name}")
                
                try:
                    if choice == 4:  # Evaluation step
                        step_func()
                    else:
                        step_func()
                    print(f"✅ {step_name} completed successfully!")
                    break  # Exit the loop after successful completion
                except Exception as e:
                    logger.error(f"{step_name} failed: {e}")
                    print(f"❌ {step_name} failed: {e}")
                    print("Would you like to try a different step?")
                    continue
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 7.")
                continue
                
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")
            continue
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            print(f"❌ Pipeline execution failed: {e}")
            print("Would you like to try again?")
            continue


def run_automated_pipeline():
    """Run the complete pipeline automatically without user interaction."""
    print("🚀 Starting automated pipeline execution...")
    
    # Step 1: Data Generation (auto-confirm)
    print("\n📊 Step 1/5: Data Generation")
    if not run_data_generation(auto_confirm=True):
        print("❌ Data generation failed. Pipeline stopped.")
        return False
    
    # Step 2: Preprocessing (auto-confirm)
    print("\n🔧 Step 2/5: Data Preprocessing")
    if not run_preprocessing(auto_confirm=True):
        print("❌ Preprocessing failed. Pipeline stopped.")
        return False
    
    # Step 3: Training (auto-confirm)
    print("\n🤖 Step 3/5: Model Training")
    model_path = run_training(auto_confirm=True)
    if not model_path:
        print("❌ Training failed. Pipeline stopped.")
        return False
    
    # Step 4: Evaluation (auto-confirm)
    print("\n📈 Step 4/5: Model Evaluation")
    if not run_evaluation(model_path, "data/featurized"):
        print("❌ Evaluation failed. Pipeline stopped.")
        return False
    
    # Step 5: Development Server (skip in training container)
    print("\n🌐 Step 5/5: Development Server")
    print("ℹ️  Skipping development server in training container.")
    print("   To run the API server, use: docker-compose --profile serve up")
    
    print("\n🎉 Complete pipeline executed successfully!")
    print("📊 Next: Start the API server with: docker-compose --profile serve up")
    return True

if __name__ == "__main__":
    main()
