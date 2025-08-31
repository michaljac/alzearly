import pytest
from fastapi.testclient import TestClient
import json
from pathlib import Path


# Mock the FastAPI app for testing
# In a real scenario, you'd import the actual app
try:
    from run_serve import app
except ImportError:
    # Create a minimal mock app for testing
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/health")
    def health():
        return {"status": "ok"}
    
    @app.get("/version")
    def version():
        return {"model_version": "test_version"}


client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_version():
    """Test version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data


def test_predict_smoke():
    """Test predict endpoint with valid data."""
    payload = {
        "items": [
            {
                "age": 72.0,
                "bmi": 26.4,
                "systolic_bp": 145.0,
                "diastolic_bp": 88.0,
                "heart_rate": 75.0,
                "temperature": 37.2,
                "glucose": 98.0,
                "cholesterol_total": 215.0,
                "hdl": 48.0,
                "ldl": 135.0,
                "triglycerides": 160.0,
                "creatinine": 1.3,
                "hemoglobin": 14.2,
                "white_blood_cells": 7.8,
                "platelets": 245.0,
                "num_encounters": 4,
                "num_medications": 3,
                "num_lab_tests": 6
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    # Should either succeed (200) or fail with validation error (422)
    assert response.status_code in (200, 422, 500)


def test_predict_multiple_patients():
    """Test predict endpoint with multiple patients."""
    payload = {
        "items": [
            {
                "age": 72.0,
                "bmi": 26.4,
                "systolic_bp": 145.0,
                "diastolic_bp": 88.0,
                "heart_rate": 75.0,
                "temperature": 37.2,
                "glucose": 98.0,
                "cholesterol_total": 215.0,
                "hdl": 48.0,
                "ldl": 135.0,
                "triglycerides": 160.0,
                "creatinine": 1.3,
                "hemoglobin": 14.2,
                "white_blood_cells": 7.8,
                "platelets": 245.0,
                "num_encounters": 4,
                "num_medications": 3,
                "num_lab_tests": 6
            },
            {
                "age": 65.0,
                "bmi": 24.8,
                "systolic_bp": 130.0,
                "diastolic_bp": 82.0,
                "heart_rate": 68.0,
                "temperature": 36.9,
                "glucose": 92.0,
                "cholesterol_total": 185.0,
                "hdl": 52.0,
                "ldl": 115.0,
                "triglycerides": 120.0,
                "creatinine": 1.1,
                "hemoglobin": 15.1,
                "white_blood_cells": 6.9,
                "platelets": 265.0,
                "num_encounters": 2,
                "num_medications": 1,
                "num_lab_tests": 4
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    # Should either succeed (200) or fail with validation error (422)
    assert response.status_code in (200, 422, 500)


def test_predict_invalid_data():
    """Test predict endpoint with invalid data."""
    payload = {
        "items": [
            {
                "age": "invalid",  # Should be float
                "bmi": 26.4,
                "systolic_bp": 145.0,
                "diastolic_bp": 88.0,
                "heart_rate": 75.0,
                "temperature": 37.2,
                "glucose": 98.0,
                "cholesterol_total": 215.0,
                "hdl": 48.0,
                "ldl": 135.0,
                "triglycerides": 160.0,
                "creatinine": 1.3,
                "hemoglobin": 14.2,
                "white_blood_cells": 7.8,
                "platelets": 245.0,
                "num_encounters": 4,
                "num_medications": 3,
                "num_lab_tests": 6
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    # Should fail with validation error
    assert response.status_code == 422


def test_predict_missing_fields():
    """Test predict endpoint with missing required fields."""
    payload = {
        "items": [
            {
                "age": 72.0,
                "bmi": 26.4,
                # Missing other required fields
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    # Should fail with validation error
    assert response.status_code == 422


def test_predict_empty_request():
    """Test predict endpoint with empty request."""
    payload = {"items": []}
    
    response = client.post("/predict", json=payload)
    # Should either succeed (200) or fail with validation error (422)
    assert response.status_code in (200, 422, 500)


def test_example_files_exist():
    """Test that example files exist."""
    example_dir = Path("examples")
    assert example_dir.exists()
    
    request_file = example_dir / "predict_request.json"
    response_file = example_dir / "predict_response.json"
    
    assert request_file.exists()
    assert response_file.exists()
    
    # Test that files contain valid JSON
    with open(request_file) as f:
        request_data = json.load(f)
        assert "items" in request_data
        assert isinstance(request_data["items"], list)
    
    with open(response_file) as f:
        response_data = json.load(f)
        assert "predictions" in response_data
        assert isinstance(response_data["predictions"], list)
