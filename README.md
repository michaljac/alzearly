# Alzearly 🧠

> **Early Detection of Alzheimer's Disease**

<div align="center">
  <img src="readme_images/hippo.jpeg" alt="Alzearly Logo" width="50" height="50" style="background: transparent;">
</div>

A comprehensive machine learning pipeline for early detection of Alzheimer's disease using synthetic patient data, featuring XGBoost and Logistic Regression models with automated evaluation and API serving capabilities.

## <img src="readme_images/hippo.jpeg" alt="🏗️" width="25" height="25" style="background: transparent;"> **Architecture Overview**

### **Core Components:**
- **Data Generation**: Synthetic patient data with realistic Alzheimer's disease patterns
- **Data Preprocessing**: Feature engineering, encoding, and scaling
- **Model Training**: XGBoost and Logistic Regression with hyperparameter optimization
- **Model Evaluation**: Comprehensive metrics (ROC, PR curves, confusion matrices)
- **API Serving**: FastAPI development server for real-time predictions
- **Experiment Tracking**: Weights & Biases integration with descriptive run names

### **Technology Stack:**
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Polars (Lazy), Pandas
- **API Framework**: FastAPI with Uvicorn
- **Experiment Tracking**: Weights & Biases
- **Visualization**: Matplotlib, Seaborn
- **Containerization**: Docker & Docker Compose
- **Progress Tracking**: TQDM with single-line updates

## <img src="readme_images/hippo.jpeg" alt="🚀" width="25" height="25" style="background: transparent;"> **Quick Start**


### **Option 1: Complete Pipeline (Recommended)**
```bash
python main.py
```
This launches an interactive menu where you can:
- Run individual steps
- Execute the complete pipeline automatically
- Create comparison plots
- Run tests

### **Option 2: Docker Compose (Production Ready)**

#### **Complete Pipeline with Optional API Server (Recommended)**
```bash
docker-compose --profile pipeline up
```
This will:
- ✅ Run the complete training pipeline (data generation, preprocessing, training, evaluation)
- ✅ Ask if you want to start the API server immediately after training
- ✅ If yes, start the FastAPI server on port 8000
- ✅ Provide seamless experience from training to serving

#### **Training Only**
```bash
docker-compose --profile training up
```

#### **API Server Only**
```bash
docker-compose --profile serve up
```

#### **Development with Hot Reload**
```bash
docker-compose --profile serve up --build
```

### **Option 3: Individual Commands**
```bash
# Data generation
python -m src.data_gen

# Data preprocessing
python -m src.preprocess

# Model training
python -m src.train

# Model evaluation
python -m src.evaluate

# Start API server
python -m src.serve
```

### **Training Scripts**

#### **Shell Script (Recommended)**
```bash
# Run complete pipeline
./train.sh full

# Generate data only
./train.sh data --n-patients 1000

# Train models only
./train.sh train --wandb-project my-experiment

# Evaluate a model
./train.sh evaluate models/best_model.pkl data/test_data.csv
```

#### **Python Script (More Options)**
```bash
# Run complete pipeline
python train_script.py --full-pipeline

# Run individual steps
python train_script.py --data-gen --n-patients 1000
python train_script.py --preprocess --rolling-window 3
python train_script.py --train --wandb-project my-exp
```

### **Pipeline Steps**

#### **1. Data Generation**
Generates synthetic patient-year data with realistic features for Alzheimer's prediction.

**Key Parameters:**
- `--n-patients`: Number of patients to generate
- `--years`: Years to generate (comma-separated)
- `--positive-rate`: Positive rate for target variable
- `--seed`: Random seed for reproducibility

**Example:**
```bash
./train.sh data --n-patients 5000 --years "2018,2019,2020" --positive-rate 0.15
```

#### **2. Data Preprocessing**
Preprocesses patient-year data with rolling features using Polars Lazy.

**Key Parameters:**
- `--rolling-window`: Rolling window in years
- `--chunk-size`: Chunk size for processing
- `--input-dir`: Input directory (default: data/raw)
- `--output-dir`: Output directory (default: data/featurized)

**Example:**
```bash
./train.sh preprocess --rolling-window 3 --chunk-size 1000
```

#### **3. Model Training**
Trains machine learning models for Alzheimer's prediction with hyperparameter tuning.

**Key Parameters:**
- `--max-features`: Maximum number of features
- `--handle-imbalance`: Imbalance handling method (smote, class_weight, etc.)
- `--wandb-project`: Weights & Biases project name
- `--wandb-entity`: Weights & Biases entity

**Example:**
```bash
./train.sh train --wandb-project alzheimers-prediction --handle-imbalance smote
```

#### **4. Model Evaluation**
Evaluates trained models with comprehensive metrics and visualizations.

**Required Parameters:**
- Model path (e.g., `models/best_model.pkl`)
- Data path (e.g., `data/test_data.csv`)

**Example:**
```bash
./train.sh evaluate models/best_model.pkl data/test_data.csv
```

### **Common Use Cases**

#### **Quick Experiment**
```bash
# Generate small dataset and run quick training
./train.sh data --n-patients 1000
./train.sh preprocess
./train.sh train --wandb-project quick-exp
```

#### **Production Training**
```bash
# Run complete pipeline with production settings
./train.sh full --n-patients 10000 --wandb-project production-v1
```

#### **Hyperparameter Tuning**
```bash
# Train with specific hyperparameters
./train.sh train --max-features 100 --handle-imbalance smote --wandb-project hyperopt
```

### **Expected Workflow**
```bash
# Run complete pipeline with optional API server
docker-compose --profile pipeline up

# Expected output:
🧠 Alzearly - Complete Pipeline with Optional API Server
============================================================
🚀 Starting Alzearly Training Pipeline...
[Training completes...]
🎉 Training Pipeline Completed Successfully!

🤔 Would you like to start the API server now? (y/n): y
🚀 Starting API server...
📊 API available at: http://localhost:8000
📖 Interactive docs at: http://localhost:8000/docs
```

## <img src="readme_images/hippo.jpeg" alt="📊" width="25" height="25" style="background: transparent;"> **Model Comparison and Visualization**

The pipeline automatically generates comprehensive comparison plots showing:
- **ROC Curves**: XGBoost vs Logistic Regression performance
- **Precision-Recall Curves**: Model comparison on imbalanced data
- **Confusion Matrices**: Detailed classification results
- **Feature Importance**: Key predictive features

All plots are saved in the `plots/` directory with timestamps for easy tracking.

## <img src="readme_images/hippo.jpeg" alt="🔬" width="25" height="25" style="background: transparent;"> **Experiment Tracking**

The pipeline supports multiple experiment tracking systems:

### **Weights & Biases (wandb)**
- **Cloud-based tracking** with API key authentication
- **Offline mode** for local development
- **Automatic logging** of metrics, plots, and model artifacts
- **Experiment tracking** with hyperparameters and training history

### **MLflow (Local)**
- **Local file-based tracking** stored in `./mlruns/`
- **Automatic setup** and configuration
- **Artifact management** for models and evaluation results
- **Experiment organization** with run history

### **No Tracking**
- **Skip experiment tracking** entirely for faster execution
- **Local file storage** of models and plots
- **Suitable for** development and testing

### **Setup**
When you run the pipeline, you'll be prompted to choose your preferred tracking system:
```
🔬 Experiment Tracking Setup
==================================================
Select experiment tracker:
1. Weights & Biases (wandb)
2. MLflow (local)
3. No tracking

Enter choice (1-3, default=1):
```

The system will automatically:
- Install required packages if needed
- Configure tracking settings
- Handle authentication (for wandb)
- Provide fallback options if setup fails

## <img src="readme_images/hippo.jpeg" alt="🐳" width="25" height="25" style="background: transparent;"> **Docker Architecture**

### **Services**
- **Training Service**: Handles data generation, preprocessing, training, and evaluation
- **Serving Service**: FastAPI development server for real-time predictions
- **Pipeline Service**: Complete pipeline with optional API server startup

### **Shared Volumes**
- `./models` - Trained models with descriptive names
- `./artifacts` - Evaluation results
- `./data` - Generated and processed data
- `./plots` - Visualization outputs with meaningful names
- `./config` - Configuration files
- `./src` - Source code

### **Expected Workflow**
```bash
# Run complete pipeline with optional API server
docker-compose --profile pipeline up

# Expected output:
🧠 Alzearly - Complete Pipeline with Optional API Server
============================================================
🚀 Starting Alzearly Training Pipeline...
[Training completes...]
🎉 Training Pipeline Completed Successfully!

🤔 Would you like to start the API server now? (y/n): y
🚀 Starting API server...
📊 API available at: http://localhost:8000
📖 Interactive docs at: http://localhost:8000/docs
```


## <img src="readme_images/hippo.jpeg" alt="🔧" width="25" height="25" style="background: transparent;"> **Configuration**

### **Data Generation** (`config/data_gen.yaml`)
- Patient demographics and medical history
- Disease progression patterns
- Temporal data spanning 2021-2024

### **Preprocessing** (`config/preprocess.yaml`)
- Feature engineering strategies
- Categorical encoding methods
- Data validation rules

### **Model Training** (`config/model.yaml`)
- Hyperparameter ranges for optimization
- Cross-validation settings
- Model-specific parameters

## <img src="readme_images/hippo.jpeg" alt="📈" width="25" height="25" style="background: transparent;"> **Performance Features**

### **Progress Tracking**
- Single-line TQDM progress bars throughout the pipeline
- Percentage-based updates for long-running operations
- Clean, non-verbose output with essential information only

### **Parallel Processing**
- Dask integration for large-scale data processing
- Ray for distributed model training
- Multiprocessing for CPU-intensive tasks

### **Memory Optimization**
- Lazy evaluation with Polars
- Efficient categorical encoding (max 50 columns)
- Streaming data processing for large datasets

## <img src="readme_images/hippo.jpeg" alt="🌐" width="25" height="25" style="background: transparent;"> **API Endpoints**

## <img src="readme_images/hippo.jpeg" alt="🌐" width="25" height="25" style="background: transparent;"> **API Endpoints**

When the serving container is running:

- **Health Check**: `GET /health`
- **Model Info**: `GET /models`
- **Prediction**: `POST /predict`
- **Interactive Docs**: `GET /docs`

### **Quick Start - API Server**

#### **Option 1: Using the Shell Script (Recommended)**
```bash
# Run with default settings (Python)
./serve.sh

# Run with Docker
./serve.sh docker

# Run with Docker Compose
./serve.sh compose

# Run on a different port
./serve.sh python --port 8080
```

#### **Option 2: Using the Python Script Directly**
```bash
# Run with default settings (Python)
python3 run_serve.py

# Run with Docker
python3 run_serve.py --method docker

# Run with Docker Compose
python3 run_serve.py --method compose

# Run on a different port
python3 run_serve.py --port 8080
```

### **Available Methods**

#### **1. Python (Default)**
- **Best for**: Development and testing
- **Pros**: Fast startup, easy debugging, auto-reload
- **Cons**: Requires local dependencies

#### **2. Docker**
- **Best for**: Isolated deployment
- **Pros**: Consistent environment, no local dependencies
- **Cons**: Slower startup, requires Docker

#### **3. Docker Compose**
- **Best for**: Production-like environment
- **Pros**: Full environment setup, volume mounting
- **Cons**: Most complex setup

### **Example API Usage**
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make a prediction
response = requests.post("http://localhost:8000/predict", json={
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
})

prediction = response.json()
print(f"Alzheimer's Risk: {prediction['probability']:.2%}")
```

### **API Testing**
```bash
# Test with the --test flag
./serve.sh python --test

# Or manually test
curl http://localhost:8000/health
```

### **Troubleshooting**

#### **Common Issues**
1. **"No trained models found"**
   - Make sure you've run the training pipeline first
   - Check that `artifacts/` or `models/` directory exists

2. **"Missing dependencies"**
   - Install requirements: `pip install -r requirements-serve.txt`

3. **"Docker not found"**
   - Install Docker and ensure it's running
   - For Docker Compose: install Docker Compose

4. **"Port already in use"**
   - Use a different port: `--port 8080`
   - Or stop the existing service on port 8000

## <img src="readme_images/hippo.jpeg" alt="🧪" width="25" height="25" style="background: transparent;"> **Testing**

The pipeline includes comprehensive testing:
- Unit tests for core functions
- Integration tests for data flow
- Model validation tests
- API endpoint testing

Run tests with:
```bash
python main.py  # Select option 7
```

## <img src="readme_images/hippo.jpeg" alt="🔄" width="25" height="25" style="background: transparent;"> **Development Workflow**

1. **Local Development**:
   ```bash
   python main.py  # Interactive development
   ```

2. **Containerized Development**:
   ```bash
   # Complete pipeline with optional API server
   docker-compose --profile pipeline up
   
   # Or run separately:
   # Terminal 1: Training
   docker-compose --profile training up
   
   # Terminal 2: API Server
   docker-compose --profile serve up
   ```

3. **Production Deployment**:
   ```bash
   docker-compose --profile pipeline up --build
   ```

## <img src="readme_images/hippo.jpeg" alt="🛠️" width="25" height="25" style="background: transparent;"> **Troubleshooting**

### **Docker Compose Issues**
```bash
# Port already in use
# Edit docker-compose.yml to change port: "8001:8000"

# Permission issues (Linux/Mac)
sudo chown -R $USER:$USER ./data ./models ./artifacts ./plots

# View logs
docker-compose --profile pipeline logs
docker-compose --profile serve logs -f
```

### **Common Issues**
- **No models found**: Ensure training completed successfully
- **FastAPI dependencies missing**: Use the serve container instead of training container
- **Port conflicts**: Change port mapping in docker-compose.yml

## <img src="readme_images/hippo.jpeg" alt="📊" width="25" height="25" style="background: transparent;"> **Model Performance**

The pipeline automatically evaluates models using:
- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve
- **F1 Score**: Balanced precision and recall
- **Confusion Matrix**: Detailed classification results

Results are logged to Weights & Biases and saved locally for analysis.
