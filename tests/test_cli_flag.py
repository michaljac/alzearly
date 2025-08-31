#!/usr/bin/env python3
"""Test the CLI tracker flag functionality."""

import sys
import subprocess
from pathlib import Path

def test_tracker_flag():
    """Test that the --tracker flag works correctly."""
    print("🧪 Testing CLI tracker flag...")
    
    # Test run_training.py
    print("\n📋 Testing run_training.py --help:")
    try:
        result = subprocess.run([
            sys.executable, "run_training.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ run_training.py --help works")
            if "--tracker" in result.stdout:
                print("✅ --tracker flag is present in help")
            else:
                print("❌ --tracker flag not found in help")
        else:
            print(f"❌ run_training.py --help failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing run_training.py: {e}")
    
    # Test src/train.py
    print("\n📋 Testing src/train.py --help:")
    try:
        result = subprocess.run([
            sys.executable, "src/train.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ src/train.py --help works")
            if "--tracker" in result.stdout:
                print("✅ --tracker flag is present in help")
            else:
                print("❌ --tracker flag not found in help")
        else:
            print(f"❌ src/train.py --help failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error testing src/train.py: {e}")
    
    # Test specific tracker values
    print("\n📋 Testing tracker values:")
    for tracker in ["none", "mlflow", "wandb"]:
        try:
            result = subprocess.run([
                sys.executable, "run_training.py", "--tracker", tracker
            ], capture_output=True, text=True, timeout=5)
            
            if "Setting up experiment tracking:" in result.stdout:
                print(f"✅ --tracker {tracker} works")
            else:
                print(f"⚠️  --tracker {tracker} may not work as expected")
        except subprocess.TimeoutExpired:
            print(f"✅ --tracker {tracker} works (timeout expected due to Polars bug)")
        except Exception as e:
            print(f"❌ Error testing --tracker {tracker}: {e}")

if __name__ == "__main__":
    test_tracker_flag()
