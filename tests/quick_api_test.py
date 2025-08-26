#!/usr/bin/env python3
"""
Quick API validation test.
Starts the FastAPI server locally and tests the /predict endpoint.
"""

import sys
import time
import json
import requests
import subprocess
import signal
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_artifacts_exist():
    """Check if required artifacts exist before starting server."""
    print("🔍 Checking if artifacts exist...")
    
    try:
        from src.io.paths import get_model_path, get_feature_names_path, get_threshold_path, get_metrics_path
        
        required_files = [
            ("model.pkl", get_model_path()),
            ("feature_names.json", get_feature_names_path()),
            ("threshold.json", get_threshold_path()),
            ("metrics.json", get_metrics_path())
        ]
        
        missing_files = []
        for file_name, file_path in required_files:
            if not file_path.exists():
                missing_files.append(file_name)
            else:
                print(f"✅ {file_name} exists")
        
        if missing_files:
            print(f"❌ Missing required artifacts: {missing_files}")
            print("Please run training first to generate artifacts.")
            return False
        
        print("✅ All artifacts found!")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not check artifacts: {e}")
        return True  # Continue anyway


def start_server(port: int = 8000) -> Optional[subprocess.Popen]:
    """Start the FastAPI server in a subprocess."""
    print(f"🚀 Starting server on port {port}...")
    
    try:
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.serve:app", 
            "--host", "0.0.0.0", 
            "--port", str(port),
            "--reload", "false"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        if server_process.poll() is None:
            print("✅ Server started successfully!")
            return server_process
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None


def test_health_endpoint(port: int = 8000):
    """Test the /health endpoint."""
    print(f"\n🏥 Testing /health endpoint...")
    
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Model loaded: {health_data.get('model_loaded')}")
            print(f"   Feature count: {health_data.get('feature_count')}")
            print(f"   Threshold: {health_data.get('optimal_threshold')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_predict_endpoint(port: int = 8000):
    """Test the /predict endpoint with sample data."""
    print(f"\n🔮 Testing /predict endpoint...")
    
    # Sample patient data
    sample_patient = {
        "patient_id": "TEST123",
        "sex": "M",
        "region": "California",
        "occupation": "Engineer",
        "education_level": "Bachelor's",
        "marital_status": "Married",
        "insurance_type": "Private",
        "age": 65.0,
        "bmi": 26.5,
        "systolic_bp": 140.0,
        "diastolic_bp": 85.0,
        "heart_rate": 72.0,
        "temperature": 98.6,
        "glucose": 95.0,
        "cholesterol_total": 200.0,
        "hdl": 45.0,
        "ldl": 130.0,
        "triglycerides": 150.0,
        "creatinine": 1.2,
        "hemoglobin": 14.5,
        "white_blood_cells": 7.5,
        "platelets": 250.0,
        "num_encounters": 3,
        "num_medications": 2,
        "num_lab_tests": 5
    }
    
    try:
        response = requests.post(
            f"http://localhost:{port}/predict",
            json=sample_patient,
            timeout=10
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            print("✅ Prediction successful!")
            print("📊 Prediction Results:")
            print(json.dumps(prediction_data, indent=2))
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def stop_server(server_process: Optional[subprocess.Popen]):
    """Stop the server process."""
    if server_process and server_process.poll() is None:
        print("\n🛑 Stopping server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            print("✅ Server stopped")
        except subprocess.TimeoutExpired:
            print("⚠️  Server didn't stop gracefully, forcing...")
            server_process.kill()
        except Exception as e:
            print(f"⚠️  Error stopping server: {e}")


def main():
    """Run the complete API validation test."""
    print("=" * 60)
    print("🌐 API VALIDATION TEST")
    print("=" * 60)
    
    port = 8000
    server_process = None
    
    try:
        # Step 1: Check artifacts
        if not check_artifacts_exist():
            print("\n❌ Cannot run API test - missing artifacts")
            return False
        
        # Step 2: Start server
        server_process = start_server(port)
        if not server_process:
            print("\n❌ Cannot run API test - server failed to start")
            return False
        
        # Step 3: Test endpoints
        health_ok = test_health_endpoint(port)
        predict_ok = test_predict_endpoint(port)
        
        # Step 4: Report results
        print("\n" + "=" * 60)
        if health_ok and predict_ok:
            print("🎉 API VALIDATION PASSED!")
            print("✅ Server is working correctly")
            print("✅ All endpoints are responding")
        else:
            print("❌ API VALIDATION FAILED!")
            if not health_ok:
                print("❌ Health endpoint failed")
            if not predict_ok:
                print("❌ Predict endpoint failed")
        
        print("=" * 60)
        
        return health_ok and predict_ok
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    finally:
        # Always stop server
        stop_server(server_process)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
