#!/usr/bin/env python3
"""
Top-level script to run the complete training pipeline.

This script orchestrates the entire ML pipeline from data generation to model training,
with experiment tracking integration.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_gen import generate as generate_data
from src.preprocess import preprocess as preprocess_data
from src.train import train as train_model
from utils import setup_experiment_tracker


def main():
    """Run the complete training pipeline."""
    parser = argparse.ArgumentParser(description="Run complete training pipeline")
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
    
    args = parser.parse_args()
    
    print("🧠 Alzheimer's Prediction Pipeline")
    print("=" * 60)
    
    # Check if featurized data already exists
    featurized_data_exists = Path("data/featurized").exists() and any(Path("data/featurized").iterdir())
    
    if featurized_data_exists:
        print("✅ Found existing featurized data - skipping data generation and preprocessing")
        args.skip_data_gen = True
        args.skip_preprocess = True
    
    # Setup experiment tracking
    if args.tracker is None:
        print("\n🔬 Setting up experiment tracking...")
        tracker, tracker_type = setup_experiment_tracker()
    else:
        print(f"\n🔬 Using experiment tracker: {args.tracker}")
        tracker_type = args.tracker
        tracker = None  # Will be set up in the training module
    
    # Step 1: Data Generation (if needed)
    if not args.skip_data_gen:
        print("\n📊 Step 1: Data Generation")
        print("-" * 30)
        try:
            generate_data()
            print("✅ Data generation completed")
        except Exception as e:
            print(f"❌ Data generation failed: {e}")
            return 1
    else:
        print("\n⏭️  Step 1: Data Generation (skipped)")
    
    # Step 2: Data Preprocessing (if needed)
    if not args.skip_preprocess:
        print("\n🔧 Step 2: Data Preprocessing")
        print("-" * 30)
        try:
            preprocess_data()
            print("✅ Data preprocessing completed")
        except Exception as e:
            print(f"❌ Data preprocessing failed: {e}")
            return 1
    else:
        print("\n⏭️  Step 2: Data Preprocessing (skipped)")
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Model Training")
    print("-" * 30)
    try:
        # Run training with the specified tracker
        train_model(
            config_file=args.config,
            tracker=tracker_type
        )
        
        print("✅ Model training completed")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return 1
    
    print("\n📦 Step 4: Model artifacts automatically saved to artifacts/latest/")
    print("   - model.pkl: Trained model")
    print("   - feature_names.json: Feature names")
    print("   - threshold.json: Classification threshold")
    print("   - metrics.json: Training metrics")
    
    print("\n🎉 Training pipeline completed successfully!")
    print("📤 Ready for model serving with: uvicorn src.serve:app --host 0.0.0.0 --port 8000")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
