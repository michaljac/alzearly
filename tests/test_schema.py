import pytest
from pydantic import BaseModel, ValidationError, Field
from typing import List


class Patient(BaseModel):
    """Schema for a single patient."""
    age: float = Field(..., ge=0, le=120, description="Patient age")
    bmi: float = Field(..., ge=10, le=100, description="Body mass index")
    systolic_bp: float = Field(..., ge=50, le=300, description="Systolic blood pressure")
    diastolic_bp: float = Field(..., ge=30, le=200, description="Diastolic blood pressure")
    heart_rate: float = Field(..., ge=30, le=200, description="Heart rate")
    temperature: float = Field(..., ge=35, le=42, description="Body temperature (Celsius)")
    glucose: float = Field(..., ge=20, le=1000, description="Blood glucose level")
    cholesterol_total: float = Field(..., ge=50, le=500, description="Total cholesterol")
    hdl: float = Field(..., ge=10, le=200, description="HDL cholesterol")
    ldl: float = Field(..., ge=10, le=300, description="LDL cholesterol")
    triglycerides: float = Field(..., ge=10, le=1000, description="Triglycerides")
    creatinine: float = Field(..., ge=0.1, le=20, description="Creatinine level")
    hemoglobin: float = Field(..., ge=5, le=25, description="Hemoglobin level")
    white_blood_cells: float = Field(..., ge=1, le=50, description="White blood cell count")
    platelets: float = Field(..., ge=50, le=1000, description="Platelet count")
    num_encounters: int = Field(..., ge=0, description="Number of healthcare encounters")
    num_medications: int = Field(..., ge=0, description="Number of medications")
    num_lab_tests: int = Field(..., ge=0, description="Number of lab tests")


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    items: List[Patient] = Field(..., description="List of patients to predict")


def test_schema_valid():
    """Test valid patient data."""
    valid_patient = {
        "age": 70.0,
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
    
    patient = Patient(**valid_patient)
    assert patient.age == 70.0
    assert patient.bmi == 26.5


def test_schema_invalid_age():
    """Test invalid age value."""
    invalid_patient = {
        "age": "bad",  # Should be float
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
    
    with pytest.raises(ValidationError):
        Patient(**invalid_patient)


def test_schema_invalid_range():
    """Test invalid range values."""
    invalid_patient = {
        "age": 150.0,  # Should be <= 120
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
    
    with pytest.raises(ValidationError):
        Patient(**invalid_patient)


def test_prediction_request_valid():
    """Test valid prediction request."""
    valid_request = {
        "items": [
            {
                "age": 70.0,
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
    
    request = PredictionRequest(**valid_request)
    assert len(request.items) == 1
    assert request.items[0].age == 70.0


def test_prediction_request_empty():
    """Test empty prediction request."""
    empty_request = {"items": []}
    
    request = PredictionRequest(**empty_request)
    assert len(request.items) == 0
