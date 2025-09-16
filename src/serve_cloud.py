"""
Cloud-optimized FastAPI application for Alzheimer's prediction.
Loads model artifacts from Google Cloud Storage and serves predictions.
Includes MLflow tracking for monitoring and metrics.
"""

import os
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import joblib
from google.cloud import storage
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Alzearly API - Cloud",
    description="Alzheimer's disease risk prediction API running on Google Cloud Run with MLflow tracking",
    version="2.1.0"
)

# Global variables for model and config
model = None
feature_names = None
model_info = None
mlflow_client = None

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/yourusername/alzearly.mlflow")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "alzearly-production")


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    # Demographics
    age: float = 65.0
    sex: str = "F"  # "M" or "F"
    region: str = "California"
    insurance_tier: str = "Premium"
    education_level: str = "Bachelor's"
    marital_status: str = "Married"
    occupation: str = "Professional"
    
    # Healthcare utilization  
    visits_count: int = 5
    meds_count: int = 3
    labs_count: int = 8
    
    # Clinical vitals
    bmi: float = 26.5
    bp_sys: float = 135.0
    bp_dia: float = 82.0
    heart_rate: float = 72.0
    temperature: float = 98.6
    
    # Laboratory values
    hba1c: float = 5.8
    ldl: float = 125.0
    hdl: float = 48.0
    glucose: float = 95.0
    creatinine: float = 1.0
    hemoglobin: float = 13.5
    
    # Financial
    costs_total: float = 8500.0
    costs_outpatient: float = 2200.0


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    alzheimers_risk_probability: float
    risk_level: str
    prediction: str
    model_version: str
    confidence: str
    prediction_id: str


def setup_mlflow():
    """Initialize MLflow tracking."""
    global mlflow_client
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Create MLflow client
        mlflow_client = MlflowClient()
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
                logger.info(f"Created MLflow experiment: {MLFLOW_EXPERIMENT_NAME} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        except Exception as e:
            logger.warning(f"Could not create/access MLflow experiment: {e}")
        
        # Set experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        logger.info("MLflow tracking initialized successfully")
        
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}. Continuing without tracking.")
        mlflow_client = None


def log_prediction_to_mlflow(request_data: dict, prediction_result: dict, processing_time: float):
    """Log prediction to MLflow."""
    if mlflow_client is None:
        return
    
    try:
        with mlflow.start_run():
            # Log input parameters
            mlflow.log_params({
                "age": request_data["age"],
                "sex": request_data["sex"],
                "bmi": request_data["bmi"],
                "bp_sys": request_data["bp_sys"],
                "bp_dia": request_data["bp_dia"],
                "region": request_data["region"],
                "insurance_tier": request_data["insurance_tier"]
            })
            
            # Log prediction metrics
            mlflow.log_metrics({
                "alzheimers_risk_probability": prediction_result["alzheimers_risk_probability"],
                "processing_time_ms": processing_time * 1000,
                "prediction_timestamp": time.time()
            })
            
            # Log prediction metadata
            mlflow.log_params({
                "model_version": prediction_result["model_version"],
                "risk_level": prediction_result["risk_level"],
                "confidence": prediction_result["confidence"],
                "prediction_id": prediction_result["prediction_id"]
            })
            
            # Log tags
            mlflow.set_tags({
                "environment": "production",
                "service": "cloud_run",
                "api_version": "2.1.0"
            })
            
    except Exception as e:
        logger.warning(f"Failed to log to MLflow: {e}")


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
    
    # Setup MLflow
    setup_mlflow()
    
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
    mlflow_status = "connected" if mlflow_client is not None else "disconnected"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "mlflow_status": mlflow_status,
        "version": "2.1.0",
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
        "model_type": type(model).__name__ if model else None,
        "mlflow_tracking": MLFLOW_TRACKING_URI,
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME
    }


@app.get("/metrics")
async def get_metrics():
    """Get prediction metrics from MLflow."""
    if mlflow_client is None:
        raise HTTPException(status_code=503, detail="MLflow not available")
    
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return {"error": "Experiment not found"}
        
        # Get recent runs
        runs = mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=100,
            order_by=["start_time DESC"]
        )
        
        # Calculate metrics
        total_predictions = len(runs)
        high_risk_count = sum(1 for run in runs if run.data.params.get("risk_level") in ["High", "Very High"])
        avg_processing_time = np.mean([
            float(run.data.metrics.get("processing_time_ms", 0)) 
            for run in runs if run.data.metrics.get("processing_time_ms")
        ]) if runs else 0
        
        return {
            "total_predictions": total_predictions,
            "high_risk_predictions": high_risk_count,
            "high_risk_percentage": (high_risk_count / total_predictions * 100) if total_predictions > 0 else 0,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "experiment_id": experiment.experiment_id,
            "mlflow_ui": f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment.experiment_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make Alzheimer's risk prediction."""
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    prediction_id = f"pred_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"Processing prediction request {prediction_id} for patient age {request.age}")
        
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
        
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction {prediction_id} completed: risk={risk_level}, probability={probability:.4f}, time={processing_time:.3f}s")
        
        # Create response
        response = PredictionResponse(
            alzheimers_risk_probability=round(probability, 4),
            risk_level=risk_level,
            prediction=prediction,
            model_version=model_version,
            confidence=confidence,
            prediction_id=prediction_id
        )
        
        # Log to MLflow asynchronously
        try:
            log_prediction_to_mlflow(
                request.dict(), 
                response.dict(), 
                processing_time
            )
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction {prediction_id} failed: {e}")
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