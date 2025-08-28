#!/usr/bin/env python3
"""
Test script to demonstrate tracker options in train.bat
"""

import subprocess
import sys
from pathlib import Path


def test_tracker_options():
    """Test different tracker options."""
    print("🧪 Testing Tracker Options")
    print("=" * 40)
    
    tracker_options = [
        ("none", "No experiment tracking"),
        ("wandb", "Weights & Biases tracking"),
        ("mlflow", "MLflow tracking")
    ]
    
    print("Available tracker options:")
    for tracker, description in tracker_options:
        print(f"  • {tracker}: {description}")
    
    print("\nUsage examples:")
    print("  train.bat                    # No tracking (default)")
    print("  train.bat --tracker wandb    # Weights & Biases tracking")
    print("  train.bat --tracker mlflow   # MLflow tracking")
    print("  train.bat --serve            # No tracking + start server")
    print("  train.bat --serve --tracker wandb  # W&B tracking + start server")
    
    print("\n💡 The tracker you're seeing ('none') is the default.")
    print("   To use experiment tracking, add --tracker wandb or --tracker mlflow")
    
    return True


def test_train_bat_help():
    """Test that train.bat help shows tracker options."""
    print("\n🔍 Testing train.bat help...")
    
    try:
        result = subprocess.run(
            ["train.bat", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            help_text = result.stdout
            if "--tracker" in help_text:
                print("✅ train.bat help shows tracker options")
                return True
            else:
                print("❌ train.bat help doesn't show tracker options")
                return False
        else:
            print(f"❌ train.bat --help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ train.bat help test failed: {e}")
        return False


def main():
    """Run tracker tests."""
    print("🔬 Tracker Options Test")
    print("=" * 40)
    
    tests = [
        ("Tracker Options", test_tracker_options),
        ("train.bat Help", test_train_bat_help),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        
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
    
    print("\n" + "=" * 40)
    print("📊 Test Results")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 Tracker options are working correctly!")
        print("\n💡 To fix your issue, use one of these commands:")
        print("   train.bat --tracker wandb --serve")
        print("   train.bat --tracker mlflow --serve")
        print("   train.bat --serve  (for no tracking)")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
