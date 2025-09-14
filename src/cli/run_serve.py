#!/usr/bin/env python3
"""
FastAPI server for Alzheimer's disease prediction API.
"""

import argparse
import json
import pickle
import socket
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Global variables for model and metadata
model = None
feature_names = None
run_log = None
optimal_threshold = 0.5

# Pydantic models for API
class PatientData(BaseModel):
    age = Field(..., ge=0, le=120, description="Patient age")
    bmi = Field(..., ge=10, le=100, description="Body mass index")
    systolic_bp = Field(..., ge=50, le=300, description="Systolic blood pressure")
    diastolic_bp = Field(..., ge=30, le=200, description="Diastolic blood pressure")
    heart_rate = Field(..., ge=30, le=200, description="Heart rate")
    temperature = Field(..., ge=35, le=42, description="Body temperature (Celsius)")
    glucose = Field(..., ge=20, le=1000, description="Blood glucose level")
    cholesterol_total = Field(..., ge=50, le=500, description="Total cholesterol")
    hdl = Field(..., ge=10, le=200, description="HDL cholesterol")
    ldl = Field(..., ge=10, le=300, description="LDL cholesterol")
    triglycerides = Field(..., ge=10, le=1000, description="Triglycerides")
    creatinine = Field(..., ge=0.1, le=20, description="Creatinine level")
    hemoglobin = Field(..., ge=5, le=25, description="Hemoglobin level")
    white_blood_cells = Field(..., ge=1, le=50, description="White blood cell count")
    platelets = Field(..., ge=50, le=1000, description="Platelet count")
    num_encounters = Field(..., ge=0, description="Number of healthcare encounters")
    num_medications = Field(..., ge=0, description="Number of medications")
    num_lab_tests = Field(..., ge=0, description="Number of lab tests")

class PredictionRequest(BaseModel):
    items = Field(..., description="List of patient data items")

class Prediction(BaseModel):
    probability = Field(..., ge=0, le=1, description="Risk probability (0.0 = no risk, 1.0 = high risk)")
    label = Field(..., description="Prediction label (0 = low risk, 1 = high risk)")

class PredictionResponse(BaseModel):
    predictions = Field(..., description="List of predictions")

# FastAPI app
app = FastAPI(
    title="Alzheimer's Disease Prediction API",
    description="API for predicting Alzheimer's disease risk from patient clinical data",
    version="1.0.0"
)

def find_available_port(start_port=8001, max_attempts=100):
    """Find an available port starting from start_port."""
    # First check if APP_PORT environment variable is set
    import os
    env_port = os.environ.get('APP_PORT')
    if env_port:
        try:
            port = int(env_port)
            # Test if the port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except (ValueError, OSError):
            pass
    
    # Find available port
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

def load_model_and_metadata():
    """Load the trained model and metadata from artifacts/latest/ directory."""
    global model, feature_names, run_log, optimal_threshold

    artifacts_dir = Path("artifacts/latest")
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    
    print(f"🔍 Loading artifacts from: {artifacts_dir}")
    
    # Check for required files
    required_files = ["model.pkl", "feature_names.json", "run_log.json"]
    for fname in required_files:
        if not (artifacts_dir / fname).exists():
            raise FileNotFoundError(f"Missing required file: {fname} in {artifacts_dir}")

    # Load model
    with open(artifacts_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")

    # Load feature names
    with open(artifacts_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    print(f"Feature names loaded: {len(feature_names)} features")

    # Load run_log (contains threshold and metrics)
    with open(artifacts_dir / "run_log.json", "r") as f:
        run_log = json.load(f)
        optimal_threshold = run_log.get("optimal_threshold", 0.5)
    print(f"Threshold loaded: {optimal_threshold}")
    print(f"Model metadata loaded: run_name={run_log.get('run_name', 'unknown')}")

    print("All artifacts loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Load model and metadata on startup."""
    try:
        load_model_and_metadata()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Server will start but prediction endpoints will fail")

@app.get("/")
async def root():
    """API information."""
    return {
        "message": "Alzheimer's Disease Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.get("/version")
async def get_version():
    """Get model version information."""
    if run_log is None:
        return {"version": "unknown", "run_name": "unknown"}
    
    return {
        "version": "1.0.0",
        "run_name": run_log.get("run_name", "unknown"),
        "optimal_threshold": optimal_threshold,
        "feature_count": len(feature_names) if feature_names else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict Alzheimer's disease risk for patients."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.items:
        raise HTTPException(status_code=400, detail="No patient data provided")
    
    try:
        predictions = []
        
        for patient in request.items:
            # Convert patient data to feature vector
            features = [
                patient.age, patient.bmi, patient.systolic_bp, patient.diastolic_bp,
                patient.heart_rate, patient.temperature, patient.glucose,
                patient.cholesterol_total, patient.hdl, patient.ldl, patient.triglycerides,
                patient.creatinine, patient.hemoglobin, patient.white_blood_cells,
                patient.platelets, patient.num_encounters, patient.num_medications,
                patient.num_lab_tests
            ]
            
            # Make prediction
            probability = model.predict_proba([features])[0][1]
            label = 1 if probability >= optimal_threshold else 0
            
            predictions.append(Prediction(probability=float(probability), label=label))
        
        return PredictionResponse(predictions=predictions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def main():
    """Main function to run the server."""
    parser = argparse.ArgumentParser(description="Alzheimer's Disease Prediction API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Find available port if default is busy
    try:
        port = find_available_port(args.port)
        if port != args.port:
            print(f"⚠️  Port {args.port} is busy, using port {port}")
    except RuntimeError as e:
        print(f"{e}")
        return 1
    
    print(f"Starting server on {args.host}:{port}")
    print(f"API documentation: http://localhost:{port}/docs")
    print(f"Access via: http://localhost:{port}")
    
    uvicorn.run(
        "run_serve:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
