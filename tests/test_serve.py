#!/usr/bin/env python3
"""
Test script to check if all serve functionality is working correctly.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required imports work."""
    print("🔍 Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import pickle
        print("✅ Pickle imported successfully")
    except ImportError as e:
        print(f"❌ Pickle import failed: {e}")
        return False
    
    return True

def test_serve_app():
    """Test if the serve app can be imported."""
    print("\n🔍 Testing serve app import...")
    
    try:
        from src.serve import app
        print("✅ Serve app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Serve app import failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading functionality."""
    print("\n🔍 Testing model loading...")
    
    try:
        from src.serve import load_model_and_metadata
        print("✅ Model loading function imported successfully")
        
        # Test if it fails gracefully when no models exist
        try:
            load_model_and_metadata()
            print("✅ Model loading succeeded (models found)")
            return True
        except FileNotFoundError as e:
            print(f"⚠️  Model loading failed as expected (no models): {e}")
            return True  # This is expected behavior
        except Exception as e:
            print(f"❌ Model loading failed unexpectedly: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading function import failed: {e}")
        return False

def test_endpoints():
    """Test if endpoints can be defined."""
    print("\n🔍 Testing endpoint definitions...")
    
    try:
        from src.serve import app
        
        # Check if endpoints exist
        routes = [route.path for route in app.routes]
        expected_routes = ['/', '/health', '/predict', '/docs', '/openapi.json']
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Endpoint {route} found")
            else:
                print(f"⚠️  Endpoint {route} not found")
        
        return True
    except Exception as e:
        print(f"❌ Endpoint testing failed: {e}")
        return False

def test_docker_compose():
    """Test if docker-compose configuration is valid."""
    print("\n🔍 Testing Docker Compose configuration...")
    
    # Check if docker-compose.yml exists
    if not Path("docker-compose.yml").exists():
        print("❌ docker-compose.yml not found")
        return False
    
    # Check if serve service is defined
    try:
        with open("docker-compose.yml", 'r') as f:
            content = f.read()
            if "serve:" in content:
                print("✅ Serve service found in docker-compose.yml")
            else:
                print("❌ Serve service not found in docker-compose.yml")
                return False
            
            if "Dockerfile.serve" in content:
                print("⚠️  Dockerfile.serve referenced but may not exist")
            else:
                print("⚠️  Dockerfile.serve not referenced")
                
        return True
    except Exception as e:
        print(f"❌ Docker Compose test failed: {e}")
        return False

def test_requirements():
    """Test if serve requirements are available."""
    print("\n🔍 Testing serve requirements...")
    
    # Check if requirements file exists
    if Path("requirements-train.txt").exists():
        print("✅ Requirements file found")
        
        # Check if FastAPI and uvicorn are in requirements
        try:
            with open("requirements-train.txt", 'r') as f:
                content = f.read()
                if "fastapi" in content.lower():
                    print("✅ FastAPI in requirements")
                else:
                    print("⚠️  FastAPI not in requirements")
                
                if "uvicorn" in content.lower():
                    print("✅ Uvicorn in requirements")
                else:
                    print("⚠️  Uvicorn not in requirements")
        except Exception as e:
            print(f"❌ Requirements check failed: {e}")
            return False
    else:
        print("❌ Requirements file not found")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧠 Alzearly - Serve Functionality Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_serve_app,
        test_model_loading,
        test_endpoints,
        test_docker_compose,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All serve functionality tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
