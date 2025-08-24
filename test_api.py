#!/usr/bin/env python3
"""
Simple test script for the Alzheimer's Prediction API using built-in libraries.
"""

import urllib.request
import urllib.parse
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8002"

def make_request(url: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make HTTP request and return JSON response."""
    if data:
        data = json.dumps(data).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'} if data else {}
    )
    req.get_method = lambda: method
    
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = make_request(f"{BASE_URL}/health")
        print(f"✅ Health check successful!")
        print(f"Status: {response.get('status')}")
        print(f"Model loaded: {response.get('model_loaded')}")
        print(f"Features: {response.get('feature_count')}")
        print()
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print()
        return False

def test_prediction():
    """Test the prediction endpoint."""
    print("Testing prediction endpoint...")
    
    # Sample patient data
    patient_data = {
        "patient_id": "P123456",
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
        # Test with optimal threshold
        print("Testing with optimal threshold...")
        response = make_request(f"{BASE_URL}/predict", "POST", patient_data)
        print(f"✅ Prediction successful!")
        print(f"Patient ID: {response.get('patient_id')}")
        print(f"Probability: {response.get('probability'):.3f}")
        print(f"Label: {response.get('label')}")
        print(f"Threshold used: {response.get('threshold_used')}")
        print()
        
        # Test with fallback threshold
        print("Testing with fallback threshold...")
        response = make_request(f"{BASE_URL}/predict?use_fallback=true", "POST", patient_data)
        print(f"✅ Fallback prediction successful!")
        print(f"Patient ID: {response.get('patient_id')}")
        print(f"Probability: {response.get('probability'):.3f}")
        print(f"Label: {response.get('label')}")
        print(f"Threshold used: {response.get('threshold_used')}")
        print()
        return True
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        print()
        return False

def test_root():
    """Test the root endpoint."""
    print("Testing root endpoint...")
    try:
        response = make_request(f"{BASE_URL}/")
        print(f"✅ Root endpoint successful!")
        print(f"Message: {response.get('message')}")
        print(f"Version: {response.get('version')}")
        print()
        return True
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        print()
        return False

if __name__ == "__main__":
    print("Alzheimer's Prediction API Test")
    print("=" * 40)
    print()
    
    try:
        root_ok = test_root()
        health_ok = test_health()
        prediction_ok = test_prediction()
        
        print("=" * 40)
        if all([root_ok, health_ok, prediction_ok]):
            print("🎉 All tests completed successfully!")
        else:
            print("⚠️  Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure the server is running with: uvicorn src.serve:app --host 0.0.0.0 --port 8002")
