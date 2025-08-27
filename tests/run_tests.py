#!/usr/bin/env python3
"""
Test runner for the Alzheimer's prediction API project.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py test_run_serve.py  # Run specific test file
"""

import sys
import os
import unittest
import subprocess
from pathlib import Path

def run_test_file(test_file):
    """Run a specific test file."""
    print(f"🧪 Running tests from {test_file}...")
    print("=" * 60)
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run the test file
    result = subprocess.run([sys.executable, test_file], 
                          cwd=os.path.dirname(os.path.abspath(test_file)),
                          capture_output=False)
    
    return result.returncode == 0

def run_all_tests():
    """Run all test files in the tests directory."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    
    print(f"🧪 Running {len(test_files)} test files...")
    print("=" * 60)
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file.name}...")
        if run_test_file(str(test_file)):
            success_count += 1
            print(f"✅ {test_file.name} passed")
        else:
            print(f"❌ {test_file.name} failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{total_count} test files passed")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False

def main():
    """Main function to run tests."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        if not os.path.exists(test_file):
            print(f"❌ Test file {test_file} not found")
            return 1
        
        success = run_test_file(test_file)
        return 0 if success else 1
    else:
        # Run all tests
        success = run_all_tests()
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
