"""
Cloud-optimized FastAPI application for Alzheimer's prediction.
Loads model artifacts from Google Cloud Storage and serves predictions.
"""

import os
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import joblib
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Alzearly API - Cloud",
    description="Alzheimer's disease risk prediction API running on Google Cloud Run",
    version="2.0.0"
)

# Global variables for model and config
model = None
feature_names = None
model_info = None


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    # Demographics
    age: float
    sex: str  # "M" or "F"
    region: str
    insurance_tier: str
    education_level: str
    marital_status: str
    occupation: str
    
    # Healthcare utilization  
    visits_count: int
    meds_count: int
    labs_count: int
    
    # Clinical vitals
    bmi: float
    bp_sys: float
    bp_dia: float
    heart_rate: float
    temperature: float
    
    # Laboratory values
    hba1c: float
    ldl: float
    hdl: float
    glucose: float
    creatinine: float
    hemoglobin: float
    
    # Financial
    costs_total: float
    costs_outpatient: float


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    alzheimers_risk_probability: float
    risk_level: str
    prediction: str
    model_version: str
    confidence: str


def preprocess_patient_data(request: PredictionRequest, target_features: list) -> pd.DataFrame:
    """
    Transform raw patient data into the 150 features expected by the model.
    This is a simplified version of the preprocessing pipeline.
    """
    # Convert request to base features
    data = request.dict()
    
    # Map API field names to preprocessing field names
    field_mapping = {
        'bp_sys': 'systolic_bp',
        'bp_dia': 'diastolic_bp',
        'visits_count': 'num_encounters',
        'meds_count': 'num_medications',
        'labs_count': 'num_lab_tests'
    }
    
    # Apply field mapping
    for api_field, model_field in field_mapping.items():
        if api_field in data:
            data[model_field] = data.pop(api_field)
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Add missing core lab values that might be expected
    if 'cholesterol_total' not in df.columns:
        df['cholesterol_total'] = df['ldl'] + df['hdl'] + 40  # Rough estimate
    if 'triglycerides' not in df.columns:
        df['triglycerides'] = 150  # Default value
    if 'white_blood_cells' not in df.columns:
        df['white_blood_cells'] = 7.0  # Default value
    if 'platelets' not in df.columns:
        df['platelets'] = 250  # Default value
    
    # Generate engineered features
    engineered_data = {}
    
    # Base clinical features (copy existing ones)
    base_features = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature',
        'glucose', 'cholesterol_total', 'hdl', 'ldl', 'triglycerides', 'creatinine',
        'hemoglobin', 'white_blood_cells', 'platelets', 'num_encounters', 'num_medications', 'num_lab_tests'
    ]
    
    for feature in base_features:
        if feature in df.columns:
            engineered_data[feature] = df[feature].iloc[0]
    
    # Add year-related features (use current year as default)
    current_year = 2024
    engineered_data['year'] = current_year
    engineered_data['diagnosis_year_uncertainty'] = 0.0  # New patients have no uncertainty
    
    # Add patient-level aggregations (for single patient, these are just the current values)
    engineered_data['age_patient_mean'] = data['age']
    engineered_data['bmi_patient_mean'] = data['bmi'] 
    engineered_data['systolic_bp_patient_mean'] = data['systolic_bp']
    engineered_data['diastolic_bp_patient_mean'] = data['diastolic_bp']
    
    # Add rolling window features (for new patients, use current values)
    for window in [1, 2, 3]:
        for feature in ['age', 'bmi', 'systolic_bp', 'diastolic_bp']:
            if feature in engineered_data:
                engineered_data[f'{feature}_rolling_mean_{window}y'] = engineered_data[feature]
                engineered_data[f'{feature}_rolling_std_{window}y'] = 0.0  # No variation for single record
                engineered_data[f'{feature}_rolling_min_{window}y'] = engineered_data[feature]
                engineered_data[f'{feature}_rolling_max_{window}y'] = engineered_data[feature]
    
    # Add delta features (for new patients, delta is 0)
    for feature in ['age', 'bmi', 'systolic_bp', 'diastolic_bp']:
        if feature in engineered_data:
            engineered_data[f'{feature}_delta_1y'] = 0.0
            engineered_data[f'{feature}_delta_2y'] = 0.0
    
    # Add risk features
    if 'age' in engineered_data:
        age = engineered_data['age']
        if age < 50:
            engineered_data['age_group_young'] = 1
            engineered_data['age_group_middle'] = 0
            engineered_data['age_group_senior'] = 0
            engineered_data['age_group_elderly'] = 0
        elif age < 65:
            engineered_data['age_group_young'] = 0
            engineered_data['age_group_middle'] = 1
            engineered_data['age_group_senior'] = 0
            engineered_data['age_group_elderly'] = 0
        elif age < 80:
            engineered_data['age_group_young'] = 0
            engineered_data['age_group_middle'] = 0
            engineered_data['age_group_senior'] = 1
            engineered_data['age_group_elderly'] = 0
        else:
            engineered_data['age_group_young'] = 0
            engineered_data['age_group_middle'] = 0
            engineered_data['age_group_senior'] = 0
            engineered_data['age_group_elderly'] = 1
    
    # Encode categorical features using one-hot encoding
    categorical_fields = ['sex', 'region', 'insurance_tier', 'education_level', 'marital_status', 'occupation']
    
    for field in categorical_fields:
        if field in data:
            # Simple one-hot encoding for common values
            if field == 'sex':
                engineered_data['sex_F'] = 1 if data[field] == 'F' else 0
                engineered_data['sex_M'] = 1 if data[field] == 'M' else 0
            else:
                # For other categorical fields, create a binary feature
                engineered_data[f'{field}_{data[field]}'] = 1
    
    # Fill missing features with default values
    for feature in target_features:
        if feature not in engineered_data:
            # Set default values based on feature type
            if 'age' in feature or 'year' in feature:
                engineered_data[feature] = data.get('age', 65)
            elif 'bmi' in feature:
                engineered_data[feature] = data.get('bmi', 25)
            elif 'bp' in feature or 'blood' in feature:
                engineered_data[feature] = data.get('systolic_bp', 120) if 'systolic' in feature else data.get('diastolic_bp', 80)
            elif 'glucose' in feature:
                engineered_data[feature] = data.get('glucose', 100)
            elif 'std' in feature or 'delta' in feature:
                engineered_data[feature] = 0.0  # Default for variation/change features
            elif any(cat in feature for cat in categorical_fields):
                engineered_data[feature] = 0  # Default for categorical features
            else:
                engineered_data[feature] = 0.0  # Default for other features
    
    # Create final DataFrame with all required features
    final_df = pd.DataFrame([engineered_data])
    
    # Ensure all target features are present
    for feature in target_features:
        if feature not in final_df.columns:
            final_df[feature] = 0.0
    
    # Reorder to match expected feature order
    final_df = final_df[target_features]
    
    return final_df


def download_model_from_gcs(bucket_name: str, model_version: str = "latest") -> Dict[str, Path]:
    """Download model artifacts from Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create temporary directory for model files
        temp_dir = Path(tempfile.mkdtemp(prefix="alzearly_model_"))
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Determine model version
        if model_version == "latest":
            try:
                latest_blob = bucket.blob("models/latest.json")
                latest_info = json.loads(latest_blob.download_as_text())
                model_version = latest_info["model_version"]
                logger.info(f"Using latest model version: {model_version}")
            except Exception as e:
                logger.warning(f"Could not read latest.json, using 'latest' folder: {e}")
                model_version = "latest"
        
        # Download required artifacts
        artifacts = {
            "model": "model.pkl",
            "features": "feature_names.json",
            "metadata": "run_log.json"
        }
        
        downloaded_files = {}
        
        for artifact_key, filename in artifacts.items():
            try:
                blob_path = f"models/{model_version}/{filename}"
                blob = bucket.blob(blob_path)
                
                local_path = temp_dir / filename
                blob.download_to_filename(str(local_path))
                
                downloaded_files[artifact_key] = local_path
                logger.info(f"Downloaded {filename} from gs://{bucket_name}/{blob_path}")
                
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to download {filename}")
        
        return downloaded_files
        
    except Exception as e:
        logger.error(f"Failed to download model from GCS: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model from cloud storage")


def load_model_artifacts(artifact_paths: Dict[str, Path]) -> tuple:
    """Load model artifacts from downloaded files."""
    try:
        # Load model
        model_obj = joblib.load(artifact_paths["model"])
        logger.info("Loaded model successfully")
        
        # Load feature names
        with open(artifact_paths["features"], 'r') as f:
            features = json.load(f)
        logger.info(f"Loaded {len(features)} feature names")
        
        # Load metadata
        with open(artifact_paths["metadata"], 'r') as f:
            metadata = json.load(f)
        logger.info("Loaded model metadata")
        
        return model_obj, features, metadata
        
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model artifacts")


async def initialize_model():
    """Initialize model on startup."""
    global model, feature_names, model_info
    
    # Get configuration from environment
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("GCS_BUCKET environment variable not set")
    
    model_version = os.getenv("MODEL_VERSION", "latest")
    
    logger.info(f"Initializing model from gs://{bucket_name}/models/{model_version}/")
    
    try:
        # Download model from GCS
        artifact_paths = download_model_from_gcs(bucket_name, model_version)
        
        # Load model artifacts
        model, feature_names, model_info = load_model_artifacts(artifact_paths)
        
        logger.info("Model initialization completed successfully")
        
        # Clean up temporary files
        import shutil
        temp_dir = artifact_paths["model"].parent
        shutil.rmtree(temp_dir)
        logger.info("Cleaned up temporary files")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup."""
    await initialize_model()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not_loaded"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "version": "2.0.0",
        "environment": "cloud_run"
    }


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": model_info,
        "feature_count": len(feature_names) if feature_names else 0,
        "model_type": type(model).__name__ if model else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make Alzheimer's risk prediction."""
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing prediction request for patient age {request.age}")
        
        # Preprocess the patient data to create the 150 features
        processed_df = preprocess_patient_data(request, feature_names)
        
        logger.info(f"Generated {len(processed_df.columns)} features for prediction")
        
        # Make prediction
        probability = model.predict_proba(processed_df)[0][1]  # Probability of positive class
        prediction = "High Risk" if probability > 0.5 else "Low Risk"
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = "Very High"
        elif probability >= 0.6:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Moderate"
        elif probability >= 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        # Determine confidence
        confidence_score = max(probability, 1 - probability)
        if confidence_score >= 0.9:
            confidence = "Very High"
        elif confidence_score >= 0.8:
            confidence = "High"
        elif confidence_score >= 0.7:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        model_version = model_info.get("model_version", "unknown") if model_info else "unknown"
        
        logger.info(f"Prediction completed: risk={risk_level}, probability={probability:.4f}")
        
        return PredictionResponse(
            alzheimers_risk_probability=round(probability, 4),
            risk_level=risk_level,
            prediction=prediction,
            model_version=model_version,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload model from GCS (admin endpoint)."""
    background_tasks.add_task(initialize_model)
    return {"message": "Model reload initiated"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)