#!/usr/bin/env python3
"""
Simple test script to verify the API server is working.
"""

import requests
import json
import time

def test_server():
    """Test the API server endpoints."""
    base_url = "http://localhost:8001"
    
    print("🧪 Testing API Server...")
    print("=" * 40)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"ERROR: Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Health check failed: {e}")
        return False
    
    # Test 2: Version endpoint
    print("\n2. Testing version endpoint...")
    try:
        response = requests.get(f"{base_url}/version", timeout=5)
        if response.status_code == 200:
            print("Version check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"ERROR: Version check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Version check failed: {e}")
    
    # Test 3: Root endpoint
    print("\n3. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("Root endpoint passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"ERROR: Root endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Root endpoint failed: {e}")
    
    # Test 4: Prediction endpoint (info)
    print("\n4. Testing prediction info endpoint...")
    try:
        response = requests.get(f"{base_url}/predict", timeout=5)
        if response.status_code == 200:
            print("Prediction info endpoint passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"ERROR: Prediction info endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Prediction info endpoint failed: {e}")
    
    # Test 5: Actual prediction
    print("\n5. Testing actual prediction...")
    test_data = {
        "items": [
            {
                "age": 65.0,
                "bmi": 26.5,
                "systolic_bp": 140.0,
                "diastolic_bp": 85.0,
                "heart_rate": 72.0,
                "temperature": 37.0,
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
        ]
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            print("Prediction endpoint passed")
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)}")
        else:
            print(f"ERROR: Prediction endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Prediction endpoint failed: {e}")
    
    print("\n" + "=" * 40)
    print("Server test completed!")
    print(f"Interactive docs available at: {base_url}/docs")
    return True

if __name__ == "__main__":
    # Wait a bit for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    test_server()
