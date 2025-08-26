#!/usr/bin/env python3
"""
Quick training validation test.
Tests the artifact saving functionality without running full training.
"""

import sys
import time
import json
import pickle
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.io.paths import get_latest_artifacts_dir, get_model_path, get_feature_names_path, get_threshold_path, get_metrics_path
from src.train_utils.save_artifacts import save_all_artifacts


def create_mock_model():
    """Create a mock model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple mock model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create mock data and fit
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    return model


def test_artifact_saving():
    """Test the artifact saving functionality."""
    print("🧪 Testing artifact saving functionality...")
    
    try:
        # Create mock data
        mock_model = create_mock_model()
        mock_feature_names = [f"feature_{i}" for i in range(10)]
        mock_threshold = 0.5
        mock_metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
            "run_id": "test_validation",
            "test_type": "mock_training"
        }
        
        print("📦 Saving mock artifacts...")
        
        # Save artifacts using our helper function
        saved_paths = save_all_artifacts(
            model=mock_model,
            feature_names=mock_feature_names,
            threshold=mock_threshold,
            metrics=mock_metrics,
            model_name="test_model.pkl"
        )
        
        print("✅ Artifacts saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Artifact saving failed: {e}")
        return False


def verify_artifacts():
    """Verify that all required artifacts exist."""
    print("\n🔍 Verifying required artifacts...")
    
    required_files = [
        ("test_model.pkl", get_model_path("test_model.pkl")),
        ("feature_names.json", get_feature_names_path()),
        ("threshold.json", get_threshold_path()),
        ("metrics.json", get_metrics_path())
    ]
    
    all_exist = True
    
    for file_name, file_path in required_files:
        if file_path.exists():
            print(f"✅ {file_name} exists at: {file_path}")
            
            # Verify file content
            try:
                if file_name.endswith('.pkl'):
                    with open(file_path, 'rb') as f:
                        loaded_model = pickle.load(f)
                    print(f"   ✅ Model loaded successfully")
                elif file_name.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"   ✅ JSON data loaded: {len(data)} items")
            except Exception as e:
                print(f"   ⚠️  Could not verify content: {e}")
        else:
            print(f"❌ {file_name} missing at: {file_path}")
            all_exist = False
    
    return all_exist


def cleanup_test_artifacts():
    """Clean up test artifacts."""
    print("\n🧹 Cleaning up test artifacts...")
    
    try:
        from src.train_utils.save_artifacts import load_model, load_feature_names, load_threshold, load_metrics
        
        # Load and verify artifacts work
        model = load_model("test_model.pkl")
        feature_names = load_feature_names()
        threshold = load_threshold()
        metrics = load_metrics()
        
        print("✅ All artifacts can be loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Threshold: {threshold}")
        print(f"   Metrics keys: {list(metrics.keys())}")
        
        # Remove test artifacts
        test_files = [
            get_model_path("test_model.pkl"),
            get_feature_names_path(),
            get_threshold_path(),
            get_metrics_path()
        ]
        
        for file_path in test_files:
            if file_path.exists():
                file_path.unlink()
                print(f"✅ Removed: {file_path.name}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")
        return False


def main():
    """Run the complete artifact validation test."""
    print("=" * 60)
    print("🧪 ARTIFACT SAVING VALIDATION TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Test artifact saving
    saving_success = test_artifact_saving()
    
    if not saving_success:
        print("\n❌ Artifact saving failed - cannot verify artifacts")
        return False
    
    # Step 2: Verify artifacts
    artifacts_exist = verify_artifacts()
    
    # Step 3: Test loading and cleanup
    cleanup_success = cleanup_test_artifacts()
    
    # Step 4: Report results
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    if artifacts_exist and cleanup_success:
        print("🎉 ARTIFACT VALIDATION PASSED!")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print("✅ Artifact saving and loading works correctly")
        print("✅ All required artifacts are being saved")
        print("✅ Artifacts can be loaded successfully")
    else:
        print("❌ ARTIFACT VALIDATION FAILED!")
        if not artifacts_exist:
            print("❌ Some required artifacts are missing")
        if not cleanup_success:
            print("❌ Artifact loading or cleanup failed")
    
    print("=" * 60)
    
    return artifacts_exist and cleanup_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
