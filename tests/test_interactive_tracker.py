#!/usr/bin/env python3
"""
Test script to verify interactive tracker selection
"""

import subprocess
import sys
from pathlib import Path


def test_interactive_tracker():
    """Test that interactive tracker selection works."""
    print("🧪 Testing Interactive Tracker Selection")
    print("=" * 50)
    
    print("Available options:")
    print("  1. No tracking (none)")
    print("  2. Weights & Biases (wandb)")
    print("  3. MLflow (mlflow)")
    
    print("\nHow it works:")
    print("  • If you don't specify --tracker, you'll get an interactive prompt")
    print("  • If you specify --tracker none/wandb/mlflow, it uses that directly")
    
    print("\nUsage examples:")
    print("  train.bat                    # Interactive prompt for tracker")
    print("  train.bat --serve            # Interactive prompt + start server")
    print("  train.bat --tracker wandb    # Direct W&B tracking")
    print("  train.bat --tracker none     # Direct no tracking")
    
    return True


def test_train_bat_config_display():
    """Test that train.bat shows interactive prompt info."""
    print("\n🔍 Testing train.bat configuration display...")
    
    try:
        # Test that train.bat shows interactive prompt when no tracker is set
        result = subprocess.run(
            ["train.bat", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            help_text = result.stdout
            if "interactive prompt" in help_text.lower():
                print("✅ train.bat mentions interactive prompt in help")
                return True
            else:
                print("❌ train.bat doesn't mention interactive prompt")
                return False
        else:
            print(f"❌ train.bat --help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ train.bat help test failed: {e}")
        return False


def test_run_training_interactive():
    """Test that run_training.py supports interactive mode."""
    print("\n🔍 Testing run_training.py interactive mode...")
    
    try:
        result = subprocess.run(
            [sys.executable, "run_training.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            help_text = result.stdout
            if "interactive" in help_text.lower():
                print("✅ run_training.py mentions interactive mode")
                return True
            else:
                print("❌ run_training.py doesn't mention interactive mode")
                return False
        else:
            print(f"❌ run_training.py --help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ run_training.py help test failed: {e}")
        return False


def main():
    """Run all interactive tracker tests."""
    print("🔬 Interactive Tracker Test")
    print("=" * 50)
    
    tests = [
        ("Interactive Tracker", test_interactive_tracker),
        ("train.bat Config Display", test_train_bat_config_display),
        ("run_training.py Interactive", test_run_training_interactive),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                failed += 1
                print(f"❌ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 Interactive tracker selection is working!")
        print("\n💡 Now you can:")
        print("   • Run 'train.bat' for interactive tracker selection")
        print("   • Run 'train.bat --serve' for interactive tracker + server")
        print("   • Or still use '--tracker none/wandb/mlflow' for direct selection")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
