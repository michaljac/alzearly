#!/usr/bin/env python3
"""
Complete Alzearly Pipeline Runner

This script is a self-contained solution that:
1. Detects your environment (Docker vs local Python)
2. Handles all setup automatically
3. Runs the complete ML pipeline
4. Provides comprehensive error handling
5. Validates everything before and after execution

No additional scripts needed - everything is included here!
"""

import argparse
import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, Tuple


def detect_environment() -> Tuple[bool, str]:
    """Detect if Docker is available and determine the best execution method."""
    print("🔍 Detecting environment...")
    
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
    
    required_packages = ['pandas', 'numpy', 'sklearn', 'polars']
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


def import_modules() -> bool:
    """Import required modules with comprehensive error handling."""
    print("🔍 Importing modules...")
    
    try:
        from src.data_gen import generate as generate_data
        from src.preprocess import preprocess as preprocess_data
        from src.train import train as train_model
        from utils import setup_experiment_tracker
        print("✅ All modules imported successfully")
        return True, generate_data, preprocess_data, train_model, setup_experiment_tracker
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements-train.txt")
        return False, None, None, None, None


def run_with_docker(args) -> int:
    """Run the pipeline using Docker."""
    print("🐳 Building Docker image...")
    
    try:
        # Build Docker image
        build_cmd = ['docker', 'build', '-f', 'Dockerfile.train', '-t', 'alzearly-train', '.']
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
        if args.skip_data_gen:
            docker_cmd.append('--skip-data-gen')
        if args.skip_preprocess:
            docker_cmd.append('--skip-preprocess')
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
    
    generate_data, preprocess_data, train_model, setup_experiment_tracker = modules
    
    # Now call the main pipeline function
    return main_pipeline(args, generate_data, preprocess_data, train_model, setup_experiment_tracker)


def main_pipeline(args, generate_data, preprocess_data, train_model, setup_experiment_tracker) -> int:
    """The main pipeline logic - extracted for reuse."""
    print("🧠 Alzheimer's Prediction Pipeline")
    print("=" * 60)
    
    # Check if featurized data already exists - FIXED: More robust data detection
    featurized_dir = Path("data/featurized")
    featurized_data_exists = False
    
    if featurized_dir.exists():
        # Check for actual data files, not just empty directory
        data_files = list(featurized_dir.glob("*.parquet")) + list(featurized_dir.glob("*.csv"))
        if data_files:
            featurized_data_exists = True
            print(f"✅ Found existing featurized data ({len(data_files)} files) - skipping data generation and preprocessing")
            args.skip_data_gen = True
            args.skip_preprocess = True
        else:
            print("⚠️  Found data/featurized directory but no data files - will regenerate")
    else:
        print("📁 No existing featurized data found - will generate new data")
    
    # Setup experiment tracking
    if args.tracker is None:
        print("\n🔬 Setting up experiment tracking...")
        tracker, tracker_type = setup_experiment_tracker()
    else:
        print(f"\n🔬 Using experiment tracker: {args.tracker}")
        tracker_type = args.tracker
        tracker = None  # Will be set up in the training module
    
    # Step 1: Data Generation (if needed) - FIXED: Better error handling and directory creation
    if not args.skip_data_gen:
        print("\n📊 Step 1: Data Generation")
        print("-" * 30)
        
        # Ensure data directories exist
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/featurized").mkdir(parents=True, exist_ok=True)
        
        try:
            generate_data()
            print("✅ Data generation completed")
        except FileNotFoundError as e:
            print(f"❌ Data generation failed - missing file: {e}")
            print("💡 Check if config/data_gen.yaml exists and is valid")
            return 1
        except Exception as e:
            print(f"❌ Data generation failed: {e}")
            print("💡 Check your data generation configuration")
            return 1
    else:
        print("\n⏭️  Step 1: Data Generation (skipped)")
    
    # Step 2: Data Preprocessing (if needed) - FIXED: Better error handling
    if not args.skip_preprocess:
        print("\n🔧 Step 2: Data Preprocessing")
        print("-" * 30)
        
        # Check if raw data exists before preprocessing
        raw_data_files = list(Path("data/raw").glob("*.parquet")) + list(Path("data/raw").glob("*.csv"))
        if not raw_data_files:
            print("❌ No raw data found for preprocessing")
            print("💡 Run data generation first or check data/raw directory")
            return 1
        
        try:
            preprocess_data()
            print("✅ Data preprocessing completed")
        except Exception as e:
            print(f"❌ Data preprocessing failed: {e}")
            print("💡 Check your preprocessing configuration")
            return 1
    else:
        print("\n⏭️  Step 2: Data Preprocessing (skipped)")
    
    # Step 3: Model Training - FIXED: Better error handling and config validation
    print("\n🤖 Step 3: Model Training")
    print("-" * 30)
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {args.config}")
        print("💡 Check if the config file exists or use --config to specify a different file")
        return 1
    
    try:
        # Run training with the specified tracker
        train_model(
            config_file=args.config,
            tracker=tracker_type
        )
        
        print("✅ Model training completed")
        
    except FileNotFoundError as e:
        print(f"❌ Model training failed - missing file: {e}")
        print("💡 Check if all required data files exist")
        return 1
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        print("💡 Check your model configuration and data")
        return 1
    
    # Step 4: Verify artifacts were created - FIXED: Artifact validation
    print("\n📦 Step 4: Verifying artifacts")
    print("-" * 30)
    
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
        for file_name in missing_files:
            print(f"   - {file_name}")
        print("💡 Check the training logs for errors")
    else:
        print("\n✅ All artifacts successfully created!")
    
    print("\n🎉 Training pipeline completed successfully!")
    print("📤 Ready for model serving with: uvicorn src.serve:app --host 0.0.0.0 --port 8000")
    
    return 0


def main():
    """Complete Alzearly Pipeline Runner - Self-contained solution."""
    print("🧠 Alzearly Pipeline Runner")
    print("==========================")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Complete Alzearly Pipeline Runner - Self-contained solution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py                    # Interactive mode
  python run_training.py --tracker none     # No tracking
  python run_training.py --tracker wandb    # With Weights & Biases
  python run_training.py --skip-data-gen    # Use existing data
  python run_training.py --help             # Show this help
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
        "--skip-data-gen", 
        action="store_true",
        help="Skip data generation step"
    )
    parser.add_argument(
        "--skip-preprocess", 
        action="store_true",
        help="Skip preprocessing step"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--force-python",
        action="store_true",
        help="Force local Python execution even if Docker is available"
    )
    
    args = parser.parse_args()
    
    # Step 1: Detect environment
    use_docker, env_type = detect_environment()
    
    if env_type == "none":
        return 1
    
    # Step 2: Setup paths
    if not setup_paths():
        return 1
    
    # Step 3: Check dependencies (only for local Python)
    if not use_docker or args.force_python:
        if not check_dependencies():
            return 1
    
    # Step 4: Import modules (only for local Python)
    if not use_docker or args.force_python:
        import_success, modules = import_modules()
        if not import_success:
            return 1
    else:
        modules = None
    
    # Step 5: Run the pipeline
    if use_docker and not args.force_python:
        print("\n🚀 Using Docker for execution...")
        return run_with_docker(args)
    else:
        print("\n🚀 Using local Python for execution...")
        return run_with_python(args, modules)


if __name__ == "__main__":
    sys.exit(main())
