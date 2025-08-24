"""
FastAPI service for Alzheimer's prediction model serving.

Provides endpoints for single patient prediction and health checks.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Alzheimer's Prediction API",
    description="API for predicting Alzheimer's disease risk from patient clinical data",
    version="1.0.0"
)

# Global variables for model and metadata
model = None
feature_names = None
optimal_threshold = None
fallback_threshold = None


class PatientData(BaseModel):
    """Schema for patient input data."""
    patient_id: str = Field(..., description="Unique patient identifier")
    sex: str = Field(..., description="Patient sex (M/F)")
    region: str = Field(..., description="Geographic region")
    occupation: str = Field(..., description="Patient occupation")
    education_level: str = Field(..., description="Education level")
    marital_status: str = Field(..., description="Marital status")
    insurance_type: str = Field(..., description="Insurance type")
    age: float = Field(..., ge=0, le=120, description="Patient age")
    bmi: float = Field(..., ge=10, le=100, description="Body mass index")
    systolic_bp: float = Field(..., ge=50, le=300, description="Systolic blood pressure")
    diastolic_bp: float = Field(..., ge=30, le=200, description="Diastolic blood pressure")
    heart_rate: float = Field(..., ge=30, le=200, description="Heart rate")
    temperature: float = Field(..., ge=30, le=45, description="Body temperature")
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

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    patient_id: str = Field(..., description="Patient identifier")
    probability: float = Field(..., ge=0, le=1, description="Predicted probability of Alzheimer's")
    label: int = Field(..., description="Predicted label (0=negative, 1=positive)")
    threshold_used: str = Field(..., description="Threshold used for prediction (optimal/fallback)")
    threshold_value: float = Field(..., description="Actual threshold value used")

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P123456",
                "probability": 0.75,
                "label": 1,
                "threshold_used": "optimal",
                "threshold_value": 0.55
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    feature_count: Optional[int] = Field(None, description="Number of features in model")
    optimal_threshold: Optional[float] = Field(None, description="Optimal threshold value")
    fallback_threshold: Optional[float] = Field(None, description="Fallback threshold value")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "feature_count": 150,
                "optimal_threshold": 0.55,
                "fallback_threshold": 0.35
            }
        }


def load_model_and_metadata():
    """Load the trained model and metadata from artifacts directory."""
    global model, feature_names, optimal_threshold, fallback_threshold
    
    # Try multiple possible artifact paths
    possible_paths = [
        Path("/app/artifacts"),  # Docker container path
        Path("artifacts"),       # Relative path from workspace
        Path("../artifacts"),    # Relative path from src directory
    ]
    
    artifacts_dir = None
    for path in possible_paths:
        if path.exists():
            artifacts_dir = path
            break
    
    if artifacts_dir is None:
        raise FileNotFoundError("No artifacts directory found. Tried: /app/artifacts, artifacts, ../artifacts")
    
    # Find the most recent model directory
    model_dirs = list(artifacts_dir.glob("*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {artifacts_dir}")
    
    # Use the first available directory (you might want to implement more sophisticated selection)
    model_dir = model_dirs[0]
    logger.info(f"Loading model from: {model_dir}")
    
    # Load model
    model_path = model_dir / "xgboost.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_names = metadata.get('feature_names', [])
    else:
        logger.warning("No metadata.json found, using default feature names")
        feature_names = None
    
    # Load thresholds
    threshold_path = artifacts_dir / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
        optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
        fallback_threshold = threshold_data.get('fallback_threshold', 0.5)
    else:
        logger.warning("No threshold.json found, using default thresholds")
        optimal_threshold = 0.5
        fallback_threshold = 0.5
    
    logger.info(f"Model loaded successfully. Features: {len(feature_names) if feature_names else 'unknown'}")
    logger.info(f"Optimal threshold: {optimal_threshold}, Fallback threshold: {fallback_threshold}")


def prepare_features(patient_data: PatientData) -> np.ndarray:
    """Prepare features for prediction from patient data."""
    # Convert patient data to dictionary
    data_dict = patient_data.dict()
    
    # Create a DataFrame with the patient data
    df = pd.DataFrame([data_dict])
    
    # Handle missing values
    df = df.fillna(0)
    
    # Convert categorical columns to numeric
    categorical_cols = ['sex', 'region', 'occupation', 'education_level', 'marital_status', 'insurance_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    
    # Use only the features that were used during training
    if feature_names:
        available_features = [col for col in feature_names if col in df.columns]
        missing_features = [col for col in feature_names if col not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        X = df[available_features].values.astype(float)
    else:
        # Fallback: use all available features
        exclude_cols = ['patient_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].values.astype(float)
    
    return X


@app.on_event("startup")
async def startup_event():
    """Load model and metadata on startup."""
    try:
        load_model_and_metadata()
        logger.info("Service started successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the service and model loading information.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        feature_count=len(feature_names) if feature_names else None,
        optimal_threshold=optimal_threshold,
        fallback_threshold=fallback_threshold
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient_data: PatientData, use_fallback: bool = False):
    """
    Predict Alzheimer's disease risk for a single patient.
    
    Args:
        patient_data: Patient clinical data
        use_fallback: Whether to use fallback threshold (default: False)
    
    Returns:
        Prediction result with probability and label
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        X = prepare_features(patient_data)
        
        # Get prediction probability
        proba = model.predict_proba(X)[0, 1]
        
        # Choose threshold
        threshold = fallback_threshold if use_fallback else optimal_threshold
        threshold_name = "fallback" if use_fallback else "optimal"
        
        # Make prediction
        label = 1 if proba >= threshold else 0
        
        return PredictionResponse(
            patient_id=patient_data.patient_id,
            probability=float(proba),
            label=label,
            threshold_used=threshold_name,
            threshold_value=float(threshold)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Alzheimer's Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )