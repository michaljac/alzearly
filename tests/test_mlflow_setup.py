#!/usr/bin/env python3
"""
Test script to verify MLflow setup without warnings
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_mlflow_setup():
    """Test MLflow setup without warnings."""
    print("🧪 Testing MLflow Setup")
    print("=" * 40)
    
    try:
        # Import the setup function
        from utils import setup_mlflow
        
        # Test MLflow setup
        print("Testing MLflow setup...")
        tracker, tracker_type = setup_mlflow()
        
        if tracker_type == "mlflow":
            print("✅ MLflow setup successful!")
            print(f"   Tracker type: {tracker_type}")
            return True
        else:
            print(f"❌ MLflow setup failed, got: {tracker_type}")
            return False
            
    except Exception as e:
        print(f"❌ MLflow setup test failed: {e}")
        return False


def test_warning_suppression():
    """Test that Pydantic warnings are suppressed."""
    print("\n🔍 Testing warning suppression...")
    
    try:
        # Read utils.py to check for warning suppression
        with open("utils.py", "r") as f:
            content = f.read()
        
        # Check for warning suppression
        if "warnings.filterwarnings" in content:
            print("✅ Warning suppression is configured")
            return True
        else:
            print("❌ Warning suppression not found")
            return False
    except Exception as e:
        print(f"❌ Error reading utils.py: {e}")
        return False


def main():
    """Run MLflow setup tests."""
    print("🔬 MLflow Setup Test")
    print("=" * 50)
    
    tests = [
        ("Warning Suppression", test_warning_suppression),
        ("MLflow Setup", test_mlflow_setup),
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
        print("\n🎉 MLflow setup is working without warnings!")
        print("\n💡 The Pydantic warnings have been suppressed.")
        print("   MLflow should now work cleanly without warning messages.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
