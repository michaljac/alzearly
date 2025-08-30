#!/usr/bin/env python3
"""
Alzearly Training Pipeline Runner

This script focuses on model training only:
1. Validates that featurized data exists (created by run_datagen.py)
2. Runs model training with experiment tracking
3. Validates artifacts are created
4. Provides comprehensive error handling

Data generation is handled separately by run_datagen.py
"""

import argparse
import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Tuple

# Debugging disabled


def run_data_generation() -> bool:
    """Run data generation subprocess."""
    try:
        print("🔄 Running data generation...")
        result = subprocess.run([sys.executable, 'run_datagen.py', '--force-regen'], 
                              check=True, capture_output=False, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print("❌ Data generation subprocess failed")
        return False
    except FileNotFoundError:
        print("❌ run_datagen.py not found")
        print("💡 Make sure run_datagen.py exists in the current directory")
        return False


def detect_environment() -> Tuple[bool, str]:
    """Detect if Docker is available and determine the best execution method."""
    print("🔍 Detecting environment...")
    print()
    
    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Docker found - will use containerized execution")
            return True, "docker"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    # Check if Python is available
    try:
        result = subprocess.run([sys.executable, '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Python found - will use local execution")
            return False, "python"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    print("❌ Neither Docker nor Python found")
    print("💡 Please install either Docker or Python 3.8+")
    return False, "none"


def check_dependencies() -> bool:
    """Check if required Python dependencies are installed."""
    print("🔍 Checking dependencies...")
    print()
    
    required_packages = ['pandas', 'numpy', 'sklearn', 'xgboost']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Installing requirements...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-train.txt'], 
                         check=True, capture_output=True, text=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("💡 Please install manually: pip install -r requirements-train.txt")
            return False
    
    print("✅ All dependencies available")
    return True


def setup_paths() -> bool:
    """Setup Python paths and validate project structure."""
    print("🔍 Setting up paths...")
    print()
    
    # Add src to path - FIXED: More robust path handling
    current_dir = Path(__file__).parent.absolute()
    src_path = current_dir / "src"
    
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        print("✅ Project structure validated")
        return True
    else:
        print(f"❌ Error: src directory not found at {src_path}")
        print("💡 Make sure you're running from the project root directory")
        return False


def import_modules() -> Tuple[bool, object, object]:
    """Import required modules with comprehensive error handling."""
    print("🔍 Importing modules...")
    print()
    
    try:
        from src.train import ModelTrainer, TrainingConfig
        from utils import setup_experiment_tracker
        print("✅ Training modules imported successfully")
        return True, (ModelTrainer, TrainingConfig), setup_experiment_tracker
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements-train.txt")
        return False, None, None


def run_with_docker(args) -> int:
    """Run the pipeline using Docker."""
    print("🐳 Building Docker image...")
    
    try:
        # Build Docker image
        build_cmd = ['docker', 'build', '-f', 'Dockerfile.train', '-t', 'alzearly-train', '.', '--network=host']
        result = subprocess.run(build_cmd, check=True, capture_output=True, text=True)
        print("✅ Docker image built successfully")
        
        # Prepare Docker run command
        docker_cmd = ['docker', 'run', '-it', '--rm']
        
        # Add volume mounts for data persistence
        docker_cmd.extend([
            '-v', f"{Path.cwd()}/data:/workspace/data",
            '-v', f"{Path.cwd()}/artifacts:/workspace/artifacts",
            '-v', f"{Path.cwd()}/plots:/workspace/plots",
            '-v', f"{Path.cwd()}/mlruns:/workspace/mlruns"
        ])
        
        # Add environment variables for experiment tracking
        if os.getenv('WANDB_API_KEY'):
            docker_cmd.extend(['-e', 'WANDB_API_KEY=' + os.getenv('WANDB_API_KEY')])
        
        docker_cmd.extend(['alzearly-train', 'python', 'run_training.py'])
        
        # Add command line arguments
        if args.tracker:
            docker_cmd.extend(['--tracker', args.tracker])
        if args.config != "config/model.yaml":
            docker_cmd.extend(['--config', args.config])
        
        print("🚀 Running pipeline with Docker...")
        result = subprocess.run(docker_cmd, check=True)
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker execution failed: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Docker not found or not accessible")
        return 1


def run_with_python(args, modules) -> int:
    """Run the pipeline using local Python."""
    print("🐍 Running pipeline with local Python...")
    print()
    
    training_classes, setup_experiment_tracker = modules
    
    # Now call the main pipeline function
    return main_pipeline(args, training_classes, setup_experiment_tracker)


def main_pipeline(args, training_classes, setup_experiment_tracker) -> int:
    """The main pipeline logic - TRAINING ONLY (data generation handled by shell scripts)."""
    
    print("🤖 Alzheimer's Training Pipeline")
    print("=" * 60)
    
    print()
    
    # Check for data in multiple locations (including subdirectories)
    local_featurized_dir = Path("/Data/featurized")
    docker_featurized_dir = Path("/Data/featurized")
    
    data_files = []
    
    # Check local data first (including subdirectories)
    if local_featurized_dir.exists():
        data_files = list(local_featurized_dir.rglob("*.parquet")) + list(local_featurized_dir.rglob("*.csv"))
    
    # Check Docker data if local data not found (including subdirectories)
    if not data_files and docker_featurized_dir.exists():
        data_files = list(docker_featurized_dir.rglob("*.parquet")) + list(docker_featurized_dir.rglob("*.csv"))
    
    if not data_files:
        print("❌ No featurized data found")
        print("💡 Please run the training script: ./train.sh (Linux/Mac) or train.bat (Windows)")
        return 1
    
    print(f"✅ Found {len(data_files)} data files - proceeding with training")
    
    print()
    print()
    
    # Setup experiment tracking
    if args.tracker is None:
        print("\n🔬 Setting up experiment tracking...")
        print()
        tracker, tracker_type = setup_experiment_tracker()
    else:
        print(f"\n🔬 Using experiment tracker: {args.tracker}")
        print()
        tracker_type = args.tracker
        tracker = None  # Will be set up in the training module
    
    # Info about data generation
    print("\n⚠️  Data generation and preprocessing handled separately")
    print("💡 Use: python run_datagen.py to generate/update data")
    
    print()
    print()
    
    # Step 1: Model Training (main focus now)
    print("\n🤖 Step 1: Model Training")
    print("-" * 30)
    print()
    
    # Validate config file exists
    # Handle different types of config arguments
    if hasattr(args.config, 'default'):
        config_file = args.config.default
    elif hasattr(args.config, 'value'):
        config_file = args.config.value
    else:
        config_file = str(args.config)
    
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_file}")
        print("💡 Check if the config file exists or use --config to specify a different file")
        return 1
    
    try:
        # Create training configuration
        ModelTrainer, TrainingConfig = training_classes
        config = TrainingConfig()
        
        # Create trainer and run training
        print("🚀 Starting model training...")
        print()
        print("⏳ This may take several minutes. Progress will be shown below:")
        print()
        print()
        
        trainer = ModelTrainer(config)
        
        print("📊 Loading and preparing data...")
        print()
        results = trainer.train(run_type="initial", tracker_type=tracker_type)
        
        print()
        print()
        print("✅ Model training completed successfully!")
        print("📊 Training results saved to artifacts")
        
        print()
        print()
        
    except FileNotFoundError as e:
        print(f"❌ Model training failed - missing file: {e}")
        print("💡 Check if all required data files exist")
        return 1
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        print("💡 Check your model configuration and data")
        return 1
    
    # Step 2: Verify artifacts were created
    print("\n📦 Step 2: Verifying artifacts")
    print("-" * 30)
    print()
    
    artifacts_dir = Path("artifacts/latest")
    required_files = ["model.pkl", "feature_names.json", "threshold.json", "metrics.json"]
    missing_files = []
    
    for file_name in required_files:
        file_path = artifacts_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} artifact files are missing:")
        print()
        for file_name in missing_files:
            print(f"   - {file_name}")
        print()
        print("💡 Check the training logs for errors")
    else:
        print("\n✅ All artifacts successfully created!")
    
    print()
    print()
    print("🎉 Training pipeline completed successfully!")
    print("📤 Ready for model serving with: python run_serve.py")
    print()
    
    return 0


def main():
    """Training Pipeline Runner - Training focused (data generation separate)."""
    print("🤖 Alzearly Training Runner")
    print("===========================")
    print()
    
    # Debugging disabled
    
    # Parse command line arguments (UPDATED - removed data gen options)
    parser = argparse.ArgumentParser(
        description="Alzearly Training Pipeline - Training Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py                    # Interactive mode
  python run_training.py --tracker none     # No tracking
  python run_training.py --tracker wandb    # With Weights & Biases
  python run_training.py --help             # Show this help

Note: Data generation is now separate - use run_datagen.py first
        """
    )
    parser.add_argument(
        "--tracker", 
        type=str, 
        choices=["none", "wandb", "mlflow"], 
        default=None,
        help="Experiment tracker to use (none, wandb, mlflow). If None, will prompt interactively."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model.yaml",
        help="Training configuration file path"
    )
    parser.add_argument(
        "--force-python",
        action="store_true",
        help="Force local Python execution even if Docker is available"
    )
    parser.add_argument(
        "--auto-generate",
        action="store_true",
        help="Automatically generate data if missing (non-interactive mode)"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing data without asking (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Detect environment
    use_docker, env_type = detect_environment()
    
    if env_type == "none":
        return 1
    
    print()
    
    # Step 2: Setup paths
    if not setup_paths():
        return 1
    
    print()
    
    # Step 3: Check dependencies (only for local Python)
    if not use_docker or args.force_python:
        if not check_dependencies():
            return 1
    
    print()
    
    # Step 4: Import modules (only for local Python)
    if not use_docker or args.force_python:
        import_success, train_model, setup_experiment_tracker = import_modules()
        if not import_success:
            return 1
        modules = (train_model, setup_experiment_tracker)
    else:
        modules = None
    
    print()
    
    # Step 5: Run the pipeline
    if use_docker and not args.force_python:
        print("\n🚀 Using Docker for execution...")
        print()
        return run_with_docker(args)
    else:
        print("\n🚀 Using local Python for execution...")
        print()
        return run_with_python(args, modules)


if __name__ == "__main__":
    sys.exit(main())