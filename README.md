# Alzheimer's Prediction API

A FastAPI-based service for predicting Alzheimer's disease risk from patient clinical data using docker compose.

## Engineering Standards

This repository follows governed engineering rules documented in `docs/rules.mdc` with formal versioning (R-* v1.0+). Rule changes require version bumps and change log entries.

![Model Comparison](readme_images/model_comparison.jpeg)

*Performance comparison between XGBoost and Logistic Regression models across different metrics including accuracy, precision, recall, and F1-score.*

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Architecture & Design

### Local (Docker Compose)
- **Containers:** `alzearly-datagen`, `alzearly-train`, `alzearly-serve`
- **Flow:** data → `/Data/featurized/` → train → `artifacts/latest/` → FastAPI serve
- **Run:** `python scripts/start_compose.py`

### Cloud (GCP)
- **Region:** `europe-west4` (Netherlands) for EU locality
- **Data:** GCS for Parquet + model artifacts; BigQuery external table over Parquet
- **Compute:**  
  - Cloud Run Jobs → run **datagen** + **train**, save outputs to GCS  
  - Cloud Run Service → runs FastAPI, loads latest model from GCS, auto-scales to zero
- **Tracking:** metrics + params saved with artifacts in GCS (optional: MLflow)

**Flows:**  
- **Local:** Docker volumes hold data + artifacts, FastAPI on port 8001  
- **Cloud:** Cloud Run Jobs produce data/models in GCS → BigQuery queries data → Cloud Run serves predictions

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Pipeline Architecture & Flow

### Pipeline Overview
The Alzearly system follows this intelligent pipeline with user control:

1. **Data Generation** (`datagen` service)
   - Creates synthetic patient data in `../Data/alzearly/raw/`
   - Preprocesses data to `../Data/alzearly/featurized/`
   - **Runs when**: Data doesn't exist OR user chooses to regenerate

2. **Model Training** (`training` service)  
   - Trains ML model using featurized data
   - Saves artifacts to `artifacts/latest/`
   - **Runs when**: Model doesn't exist OR user chooses to retrain

3. **API Server** (`serve` service)
   - Loads trained model and serves FastAPI endpoints
   - **Runs when**: Both data and model exist
   - **Cannot run** if data or model are missing

### Pipeline Flow Examples

#### Scenario 1: Fresh Start (No Data/Model)
```
1. User runs startup script
2. System detects: No data, no model
3. System: "No data found. Generating data..."
4. System: "No model found. Training model..."
5. System: "Starting server..."
6. Result: ✅ Server running with fresh data and model
```

#### Scenario 2: Existing Data/Model
```
1. User runs startup script  
2. System detects: Data exists, model exists
3. System: "Data found. Regenerate? (y/n)"
4. User: "n" (no)
5. System: "Model found. Retrain? (y/n)" 
6. User: "n" (no)
7. System: "Starting server with existing data and model..."
8. Result: ✅ Server running with existing data and model
```

#### Scenario 3: Partial Data (Data exists, no model)
```
1. User runs startup script
2. System detects: Data exists, no model
3. System: "Data found. Regenerate? (y/n)"
4. User: "n" (no)
5. System: "No model found. Training model..."
6. System: "Starting server..."
7. Result: ✅ Server running with existing data and new model
```

#### Scenario 4: Cannot Serve (Missing requirements)
```
1. User runs startup script
2. System detects: No data, no model
3. System: "No data found. Generating data..."
4. System: "No model found. Training model..."
5. System: "Starting server..."
6. Result: ✅ Server running (data and model created)
```

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Quick Start Summary

<div>

**🚀 Get Started in 4 Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michaljac/alzearly.git
   cd alzearly
   ```
2. **Prerequisites**
- **Windows:**
```cmd
md ..\Data\alzearly\raw
md ..\Data\alzearly\featurized
md artifacts
```

3. **Build the image**
- **Windows:**
```bash
docker build -t alzearly:v1 -f Dockerfile .
```

4. **Start the services (Cross-platform):**

**Option A: Use the Python launcher (Recommended):**
```bash
python scripts/start_compose.py
```

**Option B: Use OS-specific scripts:**
- **Windows:**
  ```cmd
  scripts\start_compose.bat
  ```

**Re-train if needed:**
```bash
python scripts/start_compose.py --retrain
```

**💡 Port Configuration:**
The default port is 8001, but you can customize it:
- **Environment variable:** `APP_PORT=8002 python scripts/start_compose.py`
- **Config file:** Edit `config/serve.yaml` to change `app_port: 8002`
- **Auto-port discovery:** If port 8001 is busy, the system will automatically find an available port


4. **Access the API:**
   - **API Documentation:** `http://localhost:8001/docs` (or your configured port)
   - **Health Check:** `http://localhost:8001/health` (or your configured port)
   - **Predictions:** `http://localhost:8001/predict` (or your configured port)


**That's it!** The pipeline will automatically generate data, train models, and start the API server.

**💡 Automation Features:**
- **MLflow tracking** is automatically enabled (no user input required)
- **Port discovery** automatically finds available ports if defaults are busy
- **Smart data detection** with user prompts for existing data/model regeneration

</div>


### **Alternative: Run Individual Services**

**Data Generation Only:**
```bash
docker-compose --profile datagen up
```

**Training Only:**
```bash
docker-compose --profile training up
```

**Serving Only (requires trained models):**
```bash
docker-compose --profile serve up
```

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Manual Service Execution

### Run Individual Services Only
If you want to run specific parts of the pipeline manually:

#### Generate Data Only
```bash
# Create directories first
mkdir -p ../Data/alzearly/{raw,featurized}
mkdir -p artifacts

# Generate data (always creates fresh data)
docker compose run --rm datagen
```

#### Train Model Only (requires existing data)
```bash
# Train model (always creates fresh model)
docker compose run --rm training
```

#### Serve API Only (requires existing data + model)
```bash
# Start server (only works if data and model exist)
docker compose up serve
```

### Service Dependencies & Requirements
- **Training** requires: Featurized data in `../Data/alzearly/featurized/`
- **Serving** requires: Model artifacts in `artifacts/latest/`
- **Data generation** creates: Raw and featurized data directories

### ⚠️ Important Notes
- **Cannot serve** without existing data and model
- **Must generate data first** if it doesn't exist
- **Must train model** after data generation
- **User choice** determines if existing data/model should be regenerated

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Project Structure

```
alzearly/                    # Project root
├── Dockerfile               # Main application image
├── .dockerignore            # Docker exclusions
├── src/                     # Source code
│   ├── api/                # API server code
│   │   └── run_serve.py    # FastAPI server
│   ├── cli/                # Command-line tools
│   │   ├── cli.py          # Main CLI interface
│   │   ├── run_datagen.py  # Data generation
│   │   └── run_training.py # Model training
│   ├── config.py           # Configuration management
│   ├── data_gen.py         # Data generation logic
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Model training
│   ├── utils.py            # Utility functions
│   └── ...                 # Other modules
├── scripts/                 # Platform-specific scripts
│   ├── start_compose.py    # Cross-platform launcher
│   └── start_compose.bat   # Windows startup
├── config/                  # Configuration files
│   ├── data_gen.yaml       # Data generation config
│   ├── model.yaml          # Model training config
│   ├── preprocess.yaml     # Preprocessing config
│   └── serve.yaml          # Server configuration
├── tests/                   # Test suite
├── docker-compose.yml       # Service orchestration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Key Implementation Snippets

**Smart Data Detection in Pipeline:**
```python
# Automatically detects if data exists and generates if needed
def run_training_pipeline():
    """Run the complete training pipeline automatically."""
    try:
        from src.cli.run_training import main
        return main() == 0
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        return False
```


## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Manual Docker Commands (Advanced)

<div>

If you prefer to run containers individually or need custom configurations:

### **Build Images**
```bash

# Windows (without network host flag)
docker build -f Dockerfile -t alzearly:v1 .
```

### **Run Individual Services**

**Data Generation:**
```bash
docker run --rm -v "$(pwd):/workspace" -v "$(pwd)/../Data/alzearly:/Data" alzearly:v1 python src/cli/run_datagen.py
```

**Training:**
```bash
docker run --rm -v "$(pwd):/workspace" -v "$(pwd)/../Data/alzearly:/Data" alzearly:v1 python src/cli/run_training.py
```

**Serving (with port mapping):**
```bash
docker run --rm -v "$(pwd):/workspace" -v "$(pwd)/../Data/alzearly:/Data" -p 8001:8001 alzearly:v1 python src/api/run_serve.py
```

**Note:** These commands work on Windows.
Docker Compose is recommended for easier management.

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> API Endpoints

<div>

### **Main Endpoint: POST `/predict`**
**Check if a patient is at risk for Alzheimer's disease.**

**Request:**
```json
{
  "items": [
    {
      "age": 65.0, "bmi": 26.5, "systolic_bp": 140.0, "diastolic_bp": 85.0,
      "heart_rate": 72.0, "temperature": 37.0, "glucose": 95.0,
      "cholesterol_total": 200.0, "hdl": 45.0, "ldl": 130.0, "triglycerides": 150.0,
      "creatinine": 1.2, "hemoglobin": 14.5, "white_blood_cells": 7.5, "platelets": 250.0,
      "num_encounters": 3, "num_medications": 2, "num_lab_tests": 5
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "probability": 0.75,  // Risk score (0.0 = no risk, 1.0 = high risk)
      "label": 1            // 0 = low risk, 1 = high risk
    }
  ]
}
```

### **Other Endpoints:**
- **GET `/health`** - Service health check
- **GET `/version`** - Model version info
- **GET `/docs`** - Interactive API documentation
- **GET `/`** - API information

</div>
</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> ML Pipeline & Technical Decisions

<div>

### **Core Architecture:**
- **Patient-level splits** → Prevents data leakage
- **Feature engineering** → Temporal aggregations (mean, std, min, max, count)
- **Two-stage feature selection** → Variance threshold + XGBoost importance
- **Optimized models** → XGBoost (50 trees, hist method) + Logistic Regression (liblinear solver)

### **Key Design Decisions:**

**Why Class Weight Over SMOTE?**
```python
# Applied during model training - no data leakage
params['class_weight'] = 'balanced'  # For Logistic Regression
scale_pos_weight = neg_count / pos_count  # For XGBoost
```
✅ **No data leakage** - Doesn't create synthetic samples in validation/test sets  
✅ **Computational efficiency** - No additional preprocessing overhead  
✅ **Production stability** - Preserves original data distribution

**Why Patient-Level Splitting?**
```python
# Prevents data leakage by keeping all records from same patient together
train_patients, val_patients = train_test_split(
    unique_patients, test_size=0.2, stratify=patient_labels
)
```
🚫 **Prevents leakage** - Patient's future data won't leak into training set  
📊 **Realistic evaluation** - Simulates real-world deployment scenarios

**Performance Optimizations:**
- **Conditional data cleaning** → Only processes data if NaN values exist
- **Optimized hyperparameters** → 2x faster training while maintaining accuracy
- **Efficient feature selection** → Reduces training time by 50-70%

</div>
</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Testing the API

<div>

**Easiest way:** Visit `http://localhost:8001/docs` for interactive testing

**Quick test with curl:**
```bash
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json | jq .
```

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Required Patient Data Fields

<div>

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `age` | float | 0-120 | Patient age |
| `bmi` | float | 10-100 | Body mass index |
| `systolic_bp` | float | 50-300 | Systolic blood pressure |
| `diastolic_bp` | float | 30-200 | Diastolic blood pressure |
| `heart_rate` | float | 30-200 | Heart rate |
| `temperature` | float | 35-42 | Body temperature (Celsius) |
| `glucose` | float | 20-1000 | Blood glucose level |
| `cholesterol_total` | float | 50-500 | Total cholesterol |
| `hdl` | float | 10-200 | HDL cholesterol |
| `ldl` | float | 10-300 | LDL cholesterol |
| `triglycerides` | float | 10-1000 | Triglycerides |
| `creatinine` | float | 0.1-20 | Creatinine level |
| `hemoglobin` | float | 5-25 | Hemoglobin level |
| `white_blood_cells` | float | 1-50 | White blood cell count |
| `platelets` | float | 50-1000 | Platelet count |
| `num_encounters` | int | ≥0 | Number of healthcare encounters |
| `num_medications` | int | ≥0 | Number of medications |
| `num_lab_tests` | int | ≥0 | Number of lab tests |

</div>
</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Docker Setup & Requirements

<div>

### Prerequisites

* **Docker**: Engine ≥ **20.10** and **Docker Compose v1**
* **Disk space**:

    * `alzearly:v1` ≈ **1.47 GB**
  * **Effective on-disk (with layer sharing)**: \~**2–2.5 GB** for images
  * **Data/artifacts (defaults)**: \~**1–3 GB**
  * **Recommendation**: keep **≥ 8 GB** free to be safe
* **Memory**:

  * Serving: **≥1 GB**
  * Training (XGBoost): **≥4 GB** minimum, **8 GB+** recommended
* **CPU**: **2+ cores** minimum (4+ recommended)
* **GPU (optional)**: NVIDIA GPU + NVIDIA Container Toolkit if you plan to use CUDA



### **Container Specifications**
| Container | Base Image | Purpose | Key Dependencies |
|-----------|------------|---------|------------------|
| `alzearly:v1` | Python 3.10-slim-bullseye | All services (datagen, train, serve) | pandas, polars, numpy, xgboost, sklearn, mlflow, fastapi, uvicorn |



</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Key Configuration Parameters

<div>

### **Data Generation (`config/data_gen.yaml`)**
```yaml
dataset:
  n_patients: 3000          # Number of patients to generate
  years: [2021, 2022, 2023, 2024]  # Years to generate data for
  positive_rate: 0.08       # Alzheimer's positive rate (5-10% recommended)
```

### **Model Training (`config/model.yaml`)**
```yaml
xgboost:
  n_estimators: 50          # Number of trees (optimized for speed)
  max_depth: 4              # Tree depth (prevents overfitting)
  learning_rate: 0.2        # Learning rate (faster convergence)
  tree_method: "hist"       # Tree building algorithm (2-3x faster)

class_imbalance:
  method: "class_weight"    # Options: "class_weight", "smote", "none"
```

</div>
</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Technical Deep Dive

<div>

### **Core Technologies & Libraries**
- **🤖 ML Stack**: XGBoost, Scikit-learn, NumPy/Pandas, Polars
- **📊 MLOps**: MLflow (automatically enabled), Pydantic, FastAPI
- **🐳 DevOps**: Docker, Volume Mounts, Environment Variables

### **Critical Parameters Explained**

| Parameter | Default | Impact | When to Change |
|-----------|---------|--------|----------------|
| `n_patients` | 3000 | Dataset size & training time | **Increase** for better performance<br>**Decrease** for faster iteration |
| `positive_rate` | 0.08 | Class balance | **Increase** for more positive cases<br>**Decrease** for rare disease simulation |
| `n_estimators` | 50 | Training speed vs accuracy | **Increase** for better performance<br>**Decrease** for faster training |

### **Performance vs. Accuracy Trade-offs**
```yaml
# Fast Training (current settings)
xgboost:
  n_estimators: 50      # 2x faster than default
  max_depth: 4          # Prevents overfitting
  learning_rate: 0.2    # Faster convergence

# High Accuracy (production settings)
xgboost:
  n_estimators: 200     # More trees for better performance
  max_depth: 6          # Capture more complex patterns
  learning_rate: 0.1    # More stable training
```

</div>
</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Important Notes

<div>

1. **Model Requirements**: The server requires trained model artifacts in `artifacts/latest/` directory
2. **Medical Disclaimer**: This is a research tool and should not be used for clinical diagnosis
3. **Port Conflicts**: The server automatically finds available ports to avoid conflicts

</div>
</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Troubleshooting

<div>

### **Common Issues:**
- **Port Already in Use**: Let server auto-find port or specify: `python src/api/run_serve.py --port 8005`
- **Model Not Found**: Ensure training completed and `artifacts/latest/` exists
- **YAML/PyYAML Issues**: Docker containers handle dependencies automatically

### **Docker Dependencies:**
All Python dependencies (including PyYAML) are pre-installed in Docker containers. No manual installation required.


</div>

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Service Status & Debugging

### Check Service Status
```bash
# View all services
docker compose ps

# View specific service
docker compose ps serve

# View service logs
docker compose logs serve
```

### Common Issues
- **"Service failed"**: Container may need more startup time (try increasing timeout)
- **Port conflicts**: Change port in `config/serve.yaml`
- **Missing artifacts**: Run `docker compose run --rm training` to create model
- **Cannot serve without data/model**: Must generate data and train model first

## <img src="readme_images/hippo.jpeg" width="20" height="20" style="vertical-align: middle; margin-right: 8px;"> Related Files

<div>

- `docker-compose.yml` - Cross-platform orchestration
- `Dockerfile` - Main application image
- `src/api/run_serve.py` - FastAPI server script
- `src/cli/run_training.py` - Model training script
- `src/cli/run_datagen.py` - Data generation script
- `scripts/start_compose.py` - Cross-platform launcher
- `artifacts/latest/` - Trained model and metadata
- `requirements.txt` - Python dependencies
- `config/` - Configuration files directory
- `src/config.py` - Configuration management system

</div>
</div>

</div>
</div>
