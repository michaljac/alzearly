# Cloud Deployment Guide

Complete guide for deploying Alzearly to Google Cloud Platform with artifact storage and Cloud Run serving.

## Overview

This guide covers:
1. **Model Artifact Upload** to Google Cloud Storage (GCS)
2. **FastAPI Deployment** to Google Cloud Run
3. **End-to-end Cloud Pipeline**

## Prerequisites

### 1. Google Cloud Setup
- Google Cloud Project created
- Authentication configured (see [google-cloud-setup.md](google-cloud-setup.md))

### 2. Required APIs
Enable these APIs in your project:
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 3. Local Tools
- Docker installed and running
- Google Cloud CLI configured
- Project dependencies installed

## Step 1: Upload Model Artifacts to GCS

### Check and Upload Artifacts
```bash
# Check if artifacts exist and upload to GCS
python src/cli/upload_gcs.py --project-id YOUR_PROJECT_ID

# Custom bucket name
python src/cli/upload_gcs.py \
  --project-id YOUR_PROJECT_ID \
  --bucket-name my-custom-bucket \
  --artifacts-dir artifacts/latest

# Force upload with custom version
python src/cli/upload_gcs.py \
  --project-id YOUR_PROJECT_ID \
  --model-version v1.2.3 \
  --force
```

### Expected Output
```
Alzearly Model Artifact Upload to GCS
==================================================
Artifacts directory: artifacts/latest
Project ID: your-project-id
Bucket: your-project-id-alzearly-models
Location: europe-west4

✅ Found: model.pkl
✅ Found: feature_names.json
✅ Found: run_log.json

✅ All required artifacts found in artifacts/latest

Uploading artifacts to gs://your-project-id-alzearly-models/models/20240914_143022/

✅ Uploaded: model.pkl → gs://your-project-id-alzearly-models/models/20240914_143022/model.pkl
✅ Uploaded: feature_names.json → gs://your-project-id-alzearly-models/models/20240914_143022/feature_names.json
✅ Uploaded: run_log.json → gs://your-project-id-alzearly-models/models/20240914_143022/run_log.json
✅ Updated latest model pointer: gs://your-project-id-alzearly-models/models/latest.json

🎉 SUCCESS: Uploaded 3 artifacts to GCS
```

## Step 2: Deploy FastAPI to Cloud Run

### Basic Deployment
```bash
# Deploy with default settings
python src/cli/deploy_cloud_run.py --project-id YOUR_PROJECT_ID

# Custom service name and region
python src/cli/deploy_cloud_run.py \
  --project-id YOUR_PROJECT_ID \
  --service-name alzearly-prod \
  --region europe-west1

# Skip build (use existing image)
python src/cli/deploy_cloud_run.py \
  --project-id YOUR_PROJECT_ID \
  --skip-build \
  --image-tag latest
```

### Advanced Deployment
```bash
# With custom environment variables
python src/cli/deploy_cloud_run.py \
  --project-id YOUR_PROJECT_ID \
  --service-name alzearly-api \
  --env-vars '{"MODEL_VERSION":"v1.2.3","DEBUG":"false"}'
```

### Expected Output
```
Alzearly FastAPI Cloud Run Deployment
==================================================
Project ID: your-project-id
Service: alzearly-api
Region: europe-west4
Bucket: your-project-id-alzearly-models

Checking prerequisites...
All prerequisites satisfied

Creating Dockerfile.cloudrun for Cloud Run deployment...
Created Dockerfile.cloudrun

Configuring Docker authentication...
Configured Docker for GCR

Building and pushing Docker image...
Image: gcr.io/your-project-id/alzearly-api:20240914-143045
Image pushed successfully

Deploying to Cloud Run...
Service deployed successfully

SUCCESS: FastAPI deployed to Cloud Run!

Service URL: https://alzearly-api-xxx-ew.a.run.app
Health check: https://alzearly-api-xxx-ew.a.run.app/health
API docs: https://alzearly-api-xxx-ew.a.run.app/docs
```

## Step 3: Testing the Deployed API

### Health Check
```bash
curl https://your-service-url/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "version": "1.0.0",
  "environment": "cloud_run"
}
```

### Model Information
```bash
curl https://your-service-url/model/info
```

### Make Predictions
```bash
curl -X POST "https://your-service-url/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 75,
    "sex": "F",
    "region": "California",
    "insurance_tier": "Medicare",
    "education_level": "High School",
    "marital_status": "Widowed",
    "occupation": "Retired",
    "visits_count": 8,
    "meds_count": 5,
    "labs_count": 12,
    "bmi": 28.5,
    "bp_sys": 145,
    "bp_dia": 85,
    "heart_rate": 72,
    "temperature": 98.6,
    "hba1c": 6.8,
    "ldl": 180,
    "hdl": 45,
    "glucose": 120,
    "creatinine": 1.2,
    "hemoglobin": 13.5,
    "costs_total": 15000,
    "costs_outpatient": 8500
  }'
```

## Complete End-to-End Pipeline

### Combined Script
Create a deployment script that handles both artifact upload and Cloud Run deployment:

```bash
#!/bin/bash
# deploy.sh - Complete deployment pipeline

PROJECT_ID="your-project-id"
SERVICE_NAME="alzearly-api"
REGION="europe-west4"

echo "Starting Alzearly Cloud Deployment Pipeline"

# Step 1: Upload artifacts to GCS
echo "Step 1: Uploading model artifacts..."
python src/cli/upload_gcs.py --project-id $PROJECT_ID

if [ $? -ne 0 ]; then
    echo " Artifact upload failed"
    exit 1
fi

# Step 2: Deploy to Cloud Run
echo "Step 2: Deploying to Cloud Run..."
python src/cli/deploy_cloud_run.py \
  --project-id $PROJECT_ID \
  --service-name $SERVICE_NAME \
  --region $REGION

if [ $? -ne 0 ]; then
    echo "Cloud Run deployment failed"
    exit 1
fi

echo "Deployment completed successfully!"
```

## Monitoring and Management

### View Logs
```bash
# Stream logs
gcloud logging tail "resource.type=cloud_run_revision" --project=YOUR_PROJECT_ID

# View logs in console
https://console.cloud.google.com/run/detail/REGION/SERVICE_NAME
```

### Update Service
```bash
# Deploy new version
python src/cli/deploy_cloud_run.py --project-id YOUR_PROJECT_ID --image-tag v2.0.0

# Update environment variables
gcloud run services update alzearly-api \
  --set-env-vars MODEL_VERSION=v2.0.0 \
  --region europe-west4
```

### Scale Service
```bash
# Set scaling limits
gcloud run services update alzearly-api \
  --min-instances 1 \
  --max-instances 100 \
  --region europe-west4
```

## Security Best Practices

### 1. IAM Roles
Assign minimal required permissions:
- **Cloud Run Developer** for deployment
- **Storage Object Admin** for artifact management
- **Cloud Build Editor** for CI/CD

### 2. Service Account
Create dedicated service account for Cloud Run:
```bash
gcloud iam service-accounts create alzearly-service \
  --display-name="Alzearly Cloud Run Service"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:alzearly-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

### 3. Network Security
- Use Cloud Armor for DDoS protection
- Implement API rate limiting
- Use HTTPS only (enforced by Cloud Run)

## Cost Optimization

### 1. Resource Limits
```bash
# Optimize for cost
gcloud run services update alzearly-api \
  --memory 1Gi \
  --cpu 0.5 \
  --min-instances 0 \
  --max-instances 10
```

### 2. Regional Selection
- Choose region closest to users
- Consider data residency requirements
- `europe-west4` (Netherlands) for EU users
- `us-central1` for US users


### Debug Commands
```bash
# Check service status
gcloud run services describe alzearly-api --region=europe-west4

# View service configuration
gcloud run configurations list --service=alzearly-api

# Test local build
docker build -f Dockerfile.cloudrun -t test-image .
docker run -p 8000:8000 -e GCS_BUCKET=test-bucket test-image
```

## Next Steps

1. **Set up CI/CD pipeline** using Cloud Build
2. **Implement monitoring** with Cloud Monitoring
3. **Add authentication** using Cloud IAM
4. **Configure custom domain** with Cloud DNS
5. **Implement A/B testing** for model versions

For advanced configurations, see the [Google Cloud Run Documentation](https://cloud.google.com/run/docs).

