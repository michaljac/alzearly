#!/usr/bin/env python3
"""
Alzearly Data Generation Runner

This script handles only data generation and preprocessing:
1. Generates synthetic patient data
2. Preprocesses and featurizes the data
3. Saves to data/featurized for training pipeline
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple


def setup_paths() -> bool:
    """Setup Python paths and validate project structure."""
    print("🔍 Setting up paths...")
    
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


def check_dependencies() -> bool:
    """Check if required Python dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = ['polars', 'pyarrow', 'faker', 'pandas', 'numpy']
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
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-datagen.txt'], 
                         check=True, capture_output=True, text=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("💡 Please install manually: pip install -r requirements-datagen.txt")
            return False
    
    print("✅ All dependencies available")
    return True


def import_modules() -> Tuple[bool, object, object]:
    """Import required modules for data generation."""
    print("🔍 Importing modules...")
    
    try:
        from src.data_gen import generate as generate_data
        from src.preprocess import preprocess as preprocess_data
        print("✅ Data generation modules imported successfully")
        return True, generate_data, preprocess_data
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements-datagen.txt")
        return False, None, None


def check_existing_data() -> bool:
    """Check if featurized data already exists."""
    # Check multiple possible locations for featurized data
    possible_dirs = [
        Path("data/featurized"),
        Path("/Data/featurized"),  # Docker mount location
        Path("../Data/alzearly/featurized")  # Host location
    ]
    
    for featurized_dir in possible_dirs:
        if featurized_dir.exists():
            data_files = list(featurized_dir.glob("*.parquet")) + list(featurized_dir.glob("*.csv"))
            if data_files:
                print(f"✅ Found existing featurized data in {featurized_dir} ({len(data_files)} files)")
                return True
    
    return False


def main_datagen_pipeline(args, generate_data, preprocess_data) -> int:
    """The main data generation pipeline."""
    print("📊 Alzheimer's Data Generation Pipeline")
    print("=" * 50)
    
    # Check if data already exists
    if check_existing_data() and not args.force_regen:
        print("⚠️  Featurized data already exists!")
        if args.interactive:
            response = input("Do you want to regenerate? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print("✅ Using existing data. Exiting.")
                return 0
        else:
            print("💡 Use --force-regen to regenerate data")
            print("✅ Data generation skipped - existing data found")
            return 0
    
    # Ensure data directories exist (use Docker mount paths)
    print("📁 Creating data directories...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("/Data/featurized").mkdir(parents=True, exist_ok=True)
    print("✅ Data directories ready")
    
    # Step 1: Data Generation
    if not args.skip_data_gen:
        print("📊 Step 1: Data Generation")
        try:
            # Load config for additional parameters
            try:
                from src.config import load_config
                config = load_config("data_gen")
                years = config.years
                positive_rate = config.positive_rate
                print(f"📋 Using config: years={years}, positive_rate={positive_rate}")
            except Exception as e:
                print(f"⚠️  Could not load config for additional params: {e}")
                years = [2021, 2022, 2023, 2024]
                positive_rate = 0.08
            
            generate_data(
                config_file=None,
                n_patients=args.num_patients,
                years=",".join(map(str, years)),
                positive_rate=positive_rate,
                out=args.output_dir,
                seed=args.seed
            )
            print("✅ Data generation completed")
        except Exception as e:
            print(f"❌ Data generation failed: {e}")
            return 1
    
    # Step 2: Data Preprocessing
    if not args.skip_preprocess:
        print("🔧 Step 2: Data Preprocessing")
        try:
            # Load config for preprocessing parameters
            try:
                from src.config import load_config
                config = load_config("data_gen")
                rolling_window_years = 3  # Default for rolling window
                chunk_size = config.rows_per_chunk
                print(f"📋 Using config: chunk_size={chunk_size}")
            except Exception as e:
                print(f"⚠️  Could not load config for preprocessing params: {e}")
                rolling_window_years = 3
                chunk_size = 3000
            
            # Determine featurized output directory
            # Always use /Data/featurized for Docker environment
            featurized_output_dir = "/Data/featurized"
            
            print(f"📁 Preprocessing: input={args.output_dir}, output={featurized_output_dir}")
            
            preprocess_data(
                config_file=None,
                input_dir=args.output_dir,
                output_dir=featurized_output_dir,
                rolling_window_years=rolling_window_years,
                chunk_size=chunk_size,
                seed=args.seed
            )
            print("✅ Data preprocessing completed")
        except Exception as e:
            print(f"❌ Data preprocessing failed: {e}")
            return 1
    
    print("🎉 Data generation pipeline completed successfully!")
    return 0


def main():
    """Main entry point for the data generation script."""
    parser = argparse.ArgumentParser(
        description="Alzearly Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_datagen.py                    # Generate default dataset from config (includes preprocessing)
  python run_datagen.py --num-patients 50000  # Override config with 50k patients
  python run_datagen.py --force-regen     # Regenerate existing data
  python run_datagen.py --skip-preprocess # Only generate raw data (skip preprocessing)
        """
    )
    
    # Load config first to get defaults
    try:
        from src.config import load_config
        config = load_config("data_gen")
        default_n_patients = config.n_patients
        default_output_dir = config.output_dir
        default_seed = config.seed
        print(f"📋 Loaded config: {default_n_patients} patients, output: {default_output_dir}, seed: {default_seed}")
    except Exception as e:
        print(f"⚠️  Could not load config, using fallback defaults: {e}")
        default_n_patients = 3000
        default_output_dir = "/Data/raw"
        default_seed = 42
    
    parser.add_argument(
        "--num-patients",
        type=int,
        default=default_n_patients,
        help=f"Number of patients to generate (default from config: {default_n_patients})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help=f"Output directory for raw data (default from config: {default_output_dir})"
    )
    
    parser.add_argument(
        "--data-size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Data size for preprocessing (default: medium)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help=f"Random seed for reproducibility (default from config: {default_seed})"
    )
    
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Force regeneration even if data exists"
    )
    
    parser.add_argument(
        "--skip-data-gen",
        action="store_true",
        help="Skip data generation step"
    )
    
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        default=False,
        help="Skip preprocessing step (default: False - preprocessing runs by default)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for user prompts"
    )
    
    args = parser.parse_args()
    
    # Setup and validation
    if not setup_paths():
        return 1
    
    if not check_dependencies():
        return 1
    
    success, generate_data, preprocess_data = import_modules()
    if not success:
        return 1
    
    # Run the pipeline
    return main_datagen_pipeline(args, generate_data, preprocess_data)


if __name__ == "__main__":
    exit(main())