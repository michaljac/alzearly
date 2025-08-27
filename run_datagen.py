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
    featurized_dir = Path("data/featurized")
    
    if featurized_dir.exists():
        data_files = list(featurized_dir.glob("*.parquet")) + list(featurized_dir.glob("*.csv"))
        if data_files:
            print(f"✅ Found existing featurized data ({len(data_files)} files)")
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
    
    # Ensure data directories exist
    print("📁 Creating data directories...")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/featurized").mkdir(parents=True, exist_ok=True)
    print("✅ Data directories ready")
    
    # Step 1: Data Generation
    if not args.skip_data_gen: