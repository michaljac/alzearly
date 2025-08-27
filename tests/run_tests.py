#!/usr/bin/env python3
"""
Test runner for the Alzheimer's prediction API project.

Usage:
    python run_tests.py                    # Run all tests (interactive)
    python run_tests.py test_run_serve.py  # Run specific test file
    python run_tests.py --train            # Run training tests only
    python run_tests.py --serve            # Run serving tests only
"""

import sys
import os
import unittest
import subprocess
import argparse
from pathlib import Path

def get_container_type():
    """Ask user which container they're in."""
    print("🧪 Alzearly Test Runner")
    print("=" * 50)
    print("Which container are you currently in?")
    print("1. Training container (alzearly-train)")
    print("2. Serving container (alzearly-serve)")
    print("3. Development environment")
    
    while True:
        try:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice == "1":
                return "train"
            elif choice == "2":
                return "serve"
            elif choice == "3":
                return "dev"
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n🛑 Test runner cancelled.")
            sys.exit(1)

def get_relevant_test_files(container_type):
    """Get test files relevant to the container type."""
    test_dir = Path(__file__).parent
    all_test_files = list(test_dir.glob("test_*.py"))
    
    if container_type == "train":
        # Training container: focus on training and pipeline tests
        return [f for f in all_test_files if any(keyword in f.name.lower() 
                for keyword in ["training", "train", "pipeline", "cli", "artifacts"])]
    
    elif container_type == "serve":
        # Serving container: focus on serving and API tests
        return [f for f in all_test_files if any(keyword in f.name.lower() 
                for keyword in ["serve", "api", "endpoint"])]
    
    else:  # dev environment
        # Development: run all tests
        return all_test_files

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

def run_tests_by_container(container_type):
    """Run tests based on container type."""
    test_files = get_relevant_test_files(container_type)
    
    if not test_files:
        print(f"⚠️  No relevant test files found for {container_type} container")
        return True
    
    print(f"🧪 Running {len(test_files)} test files for {container_type} container...")
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
    parser = argparse.ArgumentParser(description="Run tests for Alzearly project")
    parser.add_argument("test_file", nargs="?", help="Specific test file to run")
    parser.add_argument("--train", action="store_true", help="Run training tests only")
    parser.add_argument("--serve", action="store_true", help="Run serving tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If specific test file is provided, run it
    if args.test_file:
        if not os.path.exists(args.test_file):
            print(f"❌ Test file {args.test_file} not found")
            return 1
        
        success = run_test_file(args.test_file)
        return 0 if success else 1
    
    # If flags are provided, run accordingly
    if args.train:
        success = run_tests_by_container("train")
        return 0 if success else 1
    
    if args.serve:
        success = run_tests_by_container("serve")
        return 0 if success else 1
    
    if args.all:
        success = run_all_tests()
        return 0 if success else 1
    
    # Interactive mode: ask user which container they're in
    container_type = get_container_type()
    success = run_tests_by_container(container_type)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
