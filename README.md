# Alzearly <img src="readme_images/hippo.jpeg" alt="🧠" width="25" height="25" style="background: transparent;">

> **Early Detection of Alzheimer's Disease using Machine Learning**

A comprehensive machine learning pipeline that predicts Alzheimer's disease risk from patient health data. This project demonstrates how to build, train, and evaluate predictive models for early disease detection using synthetic patient data.

## 📋 **Submission Status**

**Python 3.10, API :8000, artifacts in ./artifacts/latest, MLflow in ./mlruns.**

### **MUST Items Checklist:**

- [ ] **Single-command run & Quick Start**
- [ ] **Tracker choice works w/o secrets**
- [ ] **Determinism & data cache**
- [ ] **FastAPI /health & /version + predict example**
- [ ] **Minimal tests (pipeline, API, serialize)**
- [ ] **Pinned deps & Docker notes**
- [ ] **Minimal CI**

## <img src="readme_images/hippo.jpeg" alt="🎯" width="20" height="20" style="background: transparent;"> **What This Project Does**

Alzheimer's disease is a progressive brain disorder that affects memory, thinking, and behavior. Early detection is crucial for better treatment outcomes. This project:

- **Generates realistic patient data** with health indicators that correlate with Alzheimer's risk
- **Trains machine learning models** to predict disease probability from patient features
- **Evaluates model performance** using medical-grade metrics
- **Provides a complete pipeline** from data generation to model deployment

## <img src="readme_images/hippo.jpeg" alt="🚀" width="20" height="20" style="background: transparent;"> **Quick Start - Run the Complete Pipeline**

### **Prerequisites**
- Python 3.10+ or Docker installed on your system
- Git for cloning the repository

### **Step-by-Step Instructions**

```bash
# 1. Clone and enter the project
git clone https://github.com/michaljac/alz-detect
cd alz-detect

# 2. Train (no external services)
python cli.py train --tracker none

# 3. Serve the API (port 8000)
uvicorn src.serve:app --port 8000
```

### **What Happens During Training**

1. **📊 Data Generation**: Creates synthetic patient data with realistic health patterns
2. **🔧 Feature Engineering**: Transforms raw data into predictive features  
3. **🤖 Model Training**: Trains XGBoost and Logistic Regression models
4. **📈 Evaluation**: Calculates performance metrics and creates visualizations
5. **💾 Artifact Export**: Saves trained models to `./artifacts/latest/` and `./artifacts/{timestamp}/`

### **Artifacts & Output**

- **Model files**: `./artifacts/latest/model.pkl`, `feature_names.json`, `threshold.json`, `metrics.json`
- **MLflow logs**: `./mlruns/` (if using `--tracker mlflow`)
- **Cache behavior**: If `./data/featurized` exists, reuses cached data (logs "cache hit")

### **Tracking Options**

| **Mode** | **Command** | **Behavior** | **Requirements** |
|----------|-------------|--------------|------------------|
| **None** | `python cli.py train --tracker none` | No tracking, fastest execution | None |
| **MLflow** | `python cli.py train --tracker mlflow` | Local tracking in `./mlruns/` | MLflow installed |
| **WandB** | `python cli.py train --tracker wandb` | Cloud tracking (if API key) or disabled mode | WandB installed |
| **WandB (with key)** | `WANDB_API_KEY=your_key python cli.py train --tracker wandb` | Full cloud tracking | WandB API key |

### **Expected Output**
```
🚀 Starting Alzearly Training Pipeline
🐍 Using local Python for execution...
🧠 Alzearly Training Pipeline
==================================================
✅ Cache hit: Found existing featurized data
📊 Step 1: Data Generation (skipped - using cached data)
🔧 Step 2: Data Preprocessing (skipped - using cached data)
🤖 Step 3: Model Training
   Tracker: none
🌱 Random seed set to: 42
✅ Model training completed
📦 Step 4: Exporting Artifacts
✅ Artifacts saved to: ./artifacts/latest
✅ Artifacts mirrored to: ./artifacts/20241201_143022
🎉 Training completed successfully!
📁 Final model path: /workspace/artifacts/latest/model.pkl
```

### **Deterministic Runs**

For reproducible results, the pipeline uses seeded random number generators:

- **Seed setting**: Automatically sets seeds for Python `random`, NumPy, XGBoost, and scikit-learn
- **Default seed**: `42` (can be overridden with `--seed <number>`)
- **Cache requirement**: Deterministic runs require using the same cached features under `./data/featurized`
- **Verification**: Running the same command twice with existing cache produces identical metrics and artifacts

**Example:**
```bash
# First run (creates cache)
python cli.py train --tracker none --rows 1000 --seed 123

# Second run (uses cache, should be identical)
python cli.py train --tracker none --rows 1000 --seed 123
```

## <img src="readme_images/hippo.jpeg" alt="🏗️" width="20" height="20" style="background: transparent;"> **Architecture Overview**

### **Pipeline Components**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Gen      │    │   Preprocessing  │    │   Model Train   │
│                 │    │                  │    │                 │
│ • Patient data  │───▶│ • Feature eng.   │───▶│ • XGBoost       │
│ • Health metrics│    │ • Rolling stats  │    │ • Logistic Reg  │
│ • Demographics  │    │ • Encoding       │    │ • Threshold opt │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Evaluation    │
                       │                 │
                       │ • ROC curves    │
                       │ • Confusion mat │
                       │ • Feature imp.  │
                       └─────────────────┘
```

### **Key Design Decisions**

#### **1. Data Generation Strategy**
- **Why synthetic data?** Real patient data is sensitive and hard to obtain
- **Realistic patterns**: Based on medical research linking health metrics to Alzheimer's risk
- **Temporal features**: Includes year-over-year changes to capture disease progression

#### **2. Model Selection**
- **XGBoost**: Handles complex non-linear relationships in health data
- **Logistic Regression**: Provides interpretable results for medical professionals
- **Ensemble approach**: Combines strengths of both models for better predictions

#### **3. Feature Engineering**
- **Rolling statistics**: Captures trends over time (e.g., 3-year averages)
- **Categorical encoding**: Converts text data (region, occupation) to numerical features
- **Feature selection**: Reduces dimensionality while preserving predictive power

#### **4. Threshold Optimization**
- **Medical context**: Balances false positives vs. false negatives
- **Cost-sensitive**: Prioritizes catching true cases over avoiding false alarms
- **ROC analysis**: Finds optimal decision boundary for clinical use

## <img src="readme_images/hippo.jpeg" alt="🔬" width="20" height="20" style="background: transparent;"> **Technical Choices Explained**

### **Health Parameters for Alzheimer's Detection**

The model uses these key health indicators, chosen based on medical research:

| **Category** | **Parameters** | **Why Important** |
|--------------|----------------|-------------------|
| **Demographics** | Age, Sex, Education, Region | Age is the strongest risk factor |
| **Vital Signs** | Blood pressure, Heart rate, BMI | Cardiovascular health affects brain function |
| **Lab Results** | Glucose, Cholesterol, Creatinine | Metabolic health correlates with cognitive decline |
| **Medical History** | Encounters, Medications, Lab tests | Healthcare utilization indicates health status |

### **Model Architecture Details**

#### **XGBoost Configuration**
```python
# Key parameters optimized for medical data
xgb_params = {
    "n_estimators": 100,      # Number of trees
    "max_depth": 6,           # Prevent overfitting
    "learning_rate": 0.1,     # Gradual learning
    "subsample": 0.8,         # Random sampling
    "colsample_bytree": 0.8,  # Feature sampling
    "eval_metric": "logloss"  # Medical-grade metric
}
```

#### **Logistic Regression**
```python
# Interpretable model for medical professionals
lr_params = {
    "random_state": 42,       # Reproducibility
    "max_iter": 1000,         # Convergence
    "class_weight": "balanced" # Handle class imbalance
}
```

### **Threshold Selection Strategy**
```python
# Medical decision making
def select_optimal_threshold(y_true, y_pred):
    """
    Find threshold that balances:
    - Sensitivity (catch true cases)
    - Specificity (avoid false alarms)
    - Medical cost considerations
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    optimal_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        
        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold
    
    return optimal_threshold
```

## <img src="readme_images/hippo.jpeg" alt="📊" width="20" height="20" style="background: transparent;"> **Key Code Areas**

### **Main Training Pipeline (`run_training.py`)**

```python
def main():
    """Orchestrates the complete ML pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=["none", "wandb", "mlflow"])
    parser.add_argument("--skip-data-gen", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    
    args = parser.parse_args()
    
    # Smart data detection - skip if already exists
    if Path("data/featurized").exists():
        args.skip_data_gen = True
        args.skip_preprocess = True
    
    # Execute pipeline steps
    if not args.skip_data_gen:
        generate_data()  # Create synthetic patient data
    
    if not args.skip_preprocess:
        preprocess_data()  # Feature engineering
    
    train_model(tracker=args.tracker)  # Train and evaluate models
```

### **Feature Engineering (`src/preprocess.py`)**

```python
def create_rolling_features(df):
    """Create temporal features for disease progression tracking."""
    numeric_cols = ["age", "bmi", "glucose", "cholesterol_total"]
    
    for col in numeric_cols:
        # 3-year rolling average
        df[f"{col}_3y_avg"] = df.groupby("patient_id")[col].rolling(3).mean()
        
        # Year-over-year change
        df[f"{col}_yoy_change"] = df.groupby("patient_id")[col].diff()
        
        # Trend direction
        df[f"{col}_trend"] = df.groupby("patient_id")[col].diff(2)
    
    return df
```

### **Model Training (`src/train.py`)**

```python
class ModelTrainer:
    def train(self, tracker_type="none"):
        """Train models with medical-grade evaluation."""
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self._prepare_data()
        
        # Train multiple models
        models = {
            "logistic_regression": LogisticRegression(**self.lr_params),
            "xgboost": XGBClassifier(**self.xgb_params)
        }
        
        results = {}
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate with medical metrics
            y_pred = model.predict_proba(X_test)[:, 1]
            results[name] = self._evaluate_model(y_test, y_pred)
        
        # Save best model and artifacts
        self._save_artifacts(results)
        return results
```

## <img src="readme_images/hippo.jpeg" alt="📈" width="20" height="20" style="background: transparent;"> **Performance Metrics & Visualizations**

### **Model Comparison Visualization**

The pipeline compares XGBoost and Logistic Regression models to demonstrate their different characteristics:

![Model Comparison](readme_images/model_comparison.jpeg)

**What this visualization shows:**
- **Probability Distributions**: How each model assigns risk probabilities to patients
- **ROC Curves**: Model performance in distinguishing between healthy and at-risk patients
- **Precision-Recall Curves**: Balance between finding true cases and avoiding false alarms
- **Key Insights**: XGBoost captures complex patterns while Logistic Regression provides interpretable results

### **Medical-Grade Evaluation**

The pipeline automatically generates these key visualizations:

#### **1. ROC Curves**
- **Purpose**: Shows model's ability to distinguish between healthy and at-risk patients
- **Interpretation**: Higher area under curve (AUC) = better performance
- **Medical threshold**: AUC > 0.8 is considered clinically useful

#### **2. Precision-Recall Curves**
- **Purpose**: Balances finding true cases vs. avoiding false alarms
- **Medical context**: High recall is crucial for disease detection
- **Threshold selection**: Optimizes for clinical decision making

#### **3. Feature Importance**
- **Purpose**: Identifies which health factors most predict Alzheimer's risk
- **Clinical value**: Helps doctors understand risk factors
- **Actionable insights**: Guides preventive care recommendations

#### **4. Confusion Matrix**
- **Purpose**: Shows true positives, false positives, true negatives, false negatives
- **Medical interpretation**: 
  - True Positives: Correctly identified at-risk patients
  - False Negatives: Missed cases (clinically serious)
  - False Positives: False alarms (less serious but costly)

## <img src="readme_images/hippo.jpeg" alt="🔧" width="20" height="20" style="background: transparent;"> **Advanced Usage**

### **Key Parameters to Modify**

> **💡 Quick Reference**: All parameters are organized by file below. Edit the corresponding YAML files to customize your pipeline.

#### **Data Generation Parameters** (`config/data_gen.yaml`)
```yaml
# Main parameters you should adjust:
data_gen:
  n_patients: 5000          # Number of patients to generate
  years: [2023, 2024, 2025] # Years of data to simulate
  positive_rate: 0.08      # Alzheimer's prevalence rate (8%)
  seed: 42                  # Random seed for reproducibility
```

#### **Model Training Parameters** (`config/model.yaml`)
```yaml
model:
  max_features: 150         # Maximum features to use
  test_size: 0.2           # Test set size (20%)
  handle_imbalance: "class_weight"  # How to handle class imbalance
  models: ["logistic_regression", "xgboost"]  # Models to train
```

#### **Preprocessing Parameters** (`config/preprocess.yaml`)
```yaml
preprocess:
  rolling_window: 3         # Years for rolling statistics
  chunk_size: 1000         # Memory management for large datasets
  numeric_columns:          # Health metrics to process
    - "age", "bmi", "glucose", "cholesterol_total"
```

### **Custom Training Configuration**

```bash
# Train with specific parameters
python run_training.py \
  --tracker wandb \
  --config config/custom_model.yaml \
  --skip-data-gen \
  --skip-preprocess

# Use existing data, custom config
python run_training.py \
  --tracker none \
  --config config/production.yaml
```

### **Parameter Modification Examples**

#### **Quick Testing (Small Dataset)**
```yaml
# config/quick_test.yaml
data_gen:
  n_patients: 1000          # Small dataset for testing
  years: [2020]             # Single year
  positive_rate: 0.08

model:
  max_features: 50          # Fewer features
  test_size: 0.2
```

#### **Production Training (Large Dataset)**
```yaml
# config/production.yaml
data_gen:
  n_patients: 50000         # Large dataset
  years: [2015, 2016, 2017, 2018, 2019, 2020]  # 6 years
  positive_rate: 0.12       # Lower prevalence

model:
  max_features: 200         # More features
  test_size: 0.2
  handle_imbalance: "smote" # Better for imbalanced data
```

#### **Research Experiment**
```yaml
# config/research.yaml
data_gen:
  n_patients: 10000
  years: [2021, 2022, 2023, 2024]  # 5 years
  positive_rate: 0.08       # Higher prevalence

preprocess:
  rolling_window: 5         # Longer trend analysis

model:
  max_features: 150
  models: ["xgboost"]       # Focus on one model
```

### **📝 When to Change Parameters - Complete Guide**

This section tells you exactly when and why to modify each parameter. All changes are made in the corresponding YAML files.

#### **File: `config/data_gen.yaml`**
| **Parameter** | **Current Value** | **When to Change** | **Recommended Range** |
|---------------|-------------------|-------------------|----------------------|
| `n_patients` | 5000 | **Increase for production** | 10,000+ for production, 1,000 for testing |
| `years` | [2021, 2022, 2023, 2024] | **Add more years** | [2015-2022] for longer disease progression |
| `positive_rate` | 0.08 
| `seed` | 42 | **For reproducibility** | Any integer (keep same for consistent results) |

#### **File: `config/model.yaml`**
| **Parameter** | **Current Value** | **When to Change** | **Recommended Range** |
|---------------|-------------------|-------------------|----------------------|
| `max_features` | 150 | **Memory issues** | 50-100 for testing, 200+ for production |
| `test_size` | 0.2 | **Standard medical** | 0.2 (20%) - standard for medical data |
| `handle_imbalance` | "class_weight" | **Better performance** | "smote" for imbalanced data, "none" for balanced |
| `models` | ["logistic_regression", "xgboost"] | **Focus on one** | ["xgboost"] for speed, add models for comparison |

#### **File: `config/preprocess.yaml`**
| **Parameter** | **Current Value** | **When to Change** | **Recommended Range** |
|---------------|-------------------|-------------------|----------------------|
| `rolling_window` | 3 | **Slower progression** | 5-7 years for slower disease progression |
| `chunk_size` | 1000 | **Memory errors** | 500 for large datasets, 2000+ for small datasets |
| `numeric_columns` | ["age", "bmi", "glucose", "cholesterol_total"] | **Add health metrics** | Add relevant health indicators |

#### **Quick Decision Guide**

**🔬 For Research/Testing:**
- `n_patients`: 1,000-5,000
- `max_features`: 50-100
- `models`: ["xgboost"] (faster)
- `rolling_window`: 3

**🏭 For Production:**
- `n_patients`: 10,000+
- `max_features`: 150-200
- `models`: ["logistic_regression", "xgboost"]
- `handle_imbalance`: "smote"

**💾 For Memory-Constrained Systems:**
- `chunk_size`: 500
- `max_features`: 50-100
- `n_patients`: 1,000-2,000

### **Experiment Tracking Options**

#### **Weights & Biases (Recommended)**
```bash
# Set API key
export WANDB_API_KEY=your_key_here

# Run with cloud tracking
docker run -it -e WANDB_API_KEY=your_key alzearly-train python run_training.py --tracker wandb
```

#### **MLflow (Local)**
```bash
# Local experiment tracking
python run_training.py --tracker mlflow

# View results
mlflow ui
```

#### **No Tracking (Fastest)**
```bash
# Skip experiment tracking for speed
python run_training.py --tracker none
```

### **Data Pipeline Options**

```bash
# Skip data generation (use existing data)
python run_training.py --skip-data-gen

# Skip preprocessing (use existing features)
python run_training.py --skip-preprocess

# Skip both (train only)
python run_training.py --skip-data-gen --skip-preprocess
```

## <img src="readme_images/hippo.jpeg" alt="🧪" width="20" height="20" style="background: transparent;"> **Testing the Pipeline**

```bash
# Run unit tests
python -m pytest tests/test_run_training.py -v

# Test specific functionality
python -m pytest tests/test_run_training.py::TestRunTraining::test_main_successful_pipeline -v
```

### **Smart Data Detection**

The script intelligently detects existing data:
```bash
✅ Found existing featurized data (3 files) - skipping data generation and preprocessing
⚠️  Found data/featurized directory but no data files - will regenerate
📁 No existing featurized data found - will generate new data
```

### **Artifact Verification**

After training, the script verifies all artifacts were created:
```bash
📦 Step 4: Verifying artifacts
✅ model.pkl
✅ feature_names.json
✅ threshold.json
✅ metrics.json
✅ All artifacts successfully created!
```

### **Performance Optimization**

- **Large datasets**: Use `--skip-data-gen` with existing data
- **Fast iteration**: Use `--tracker none` for no experiment tracking
- **Memory issues**: Reduce `max_features` in config
- **GPU training**: Modify Dockerfile to include CUDA support

## <img src="readme_images/hippo.jpeg" alt="🚀" width="20" height="20" style="background: transparent;"> **Serving Endpoint - API Deployment**

### **Quick Start - Run the API Server**

After training your models, you can deploy the prediction API:

```bash
# Option 1: Run with Python (development)
python run_serve.py

# Option 2: Run with Docker (production)
docker-compose up serve

# Option 3: Run with custom port
python run_serve.py --port 9000 --host 127.0.0.1 --reload
```

### **API Endpoints**

The FastAPI server provides these endpoints:

| **Endpoint** | **Method** | **Description** | **Example** |
|--------------|------------|-----------------|-------------|
| `/` | GET | API information | `curl http://localhost:8000/` |
| `/health` | GET | Health check | `curl http://localhost:8000/health` |
| `/predict` | POST | Make predictions | `curl -X POST http://localhost:8000/predict` |
| `/docs` | GET | Interactive API docs | Open `http://localhost:8000/docs` |

### **Making Predictions**

#### **Example Prediction Request**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P123456",
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
  }'
```

#### **Example Prediction Response**
```json
{
  "patient_id": "P123456",
  "probability": 0.75,
  "label": 1,
  "threshold_used": "optimal",
  "threshold_value": 0.55
}
```

### **Docker Deployment**

#### **Production Deployment**
```bash
# Build the serving container
docker build -f Dockerfile.serve -t alzearly-serve .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/artifacts:/app/artifacts \
  alzearly-serve
```

#### **Docker Compose (Recommended)**
```bash
# Start the serving service
docker-compose up serve

# Start with specific profile
docker-compose --profile serve up
```

### **Health Check**

Monitor the API health:
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 150,
  "optimal_threshold": 0.55,
  "fallback_threshold": 0.35
}
```

### **Interactive API Documentation**

Open your browser to `http://localhost:8000/docs` for:
- **Interactive API testing**
- **Request/response schemas**
- **Example requests**
- **Real-time API exploration**

## <img src="readme_images/hippo.jpeg" alt="🧪" width="20" height="20" style="background: transparent;"> **Testing the Serving Endpoint**

### **Unit Tests for API Server**

Comprehensive unit tests are available for the serving functionality:

```bash
# Run all serving tests
cd tests && python run_tests.py

# Run specific serving tests
cd tests && python test_run_serve.py

# Run individual test classes
cd tests && python -c "
import unittest
from test_run_serve import TestRunServe, TestRunServeIntegration
unittest.main(argv=[''], exit=False, verbosity=2)
"
```

### **Test Coverage**

The serving tests cover:

| **Test Category** | **Description** | **Test Count** |
|-------------------|-----------------|----------------|
| **Argument Parsing** | CLI argument validation | 5 tests |
| **Server Startup** | Uvicorn integration | 4 tests |
| **Error Handling** | Exception management | 3 tests |
| **Output Validation** | Console output verification | 2 tests |
| **Integration** | FastAPI app validation | 3 tests |
| **Code Quality** | Import and structure checks | 4 tests |

### **Running Tests in Docker**

```bash
# Test the serving container
docker-compose --profile test up test

# Test with specific configuration
docker run --rm -v $(pwd)/tests:/app/tests \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/artifacts:/app/artifacts \
  alzearly-serve python /app/tests/test_run_serve.py
```

### **Test Results Example**

```bash
🧪 Running tests from test_run_serve.py...
============================================================
test_argument_parser_custom_values (__main__.TestRunServe) ... ok
test_argument_parser_defaults (__main__.TestRunServe) ... ok
test_main_function_success (__main__.TestRunServe) ... ok
test_fastapi_app_import (__main__.TestRunServeIntegration) ... ok
test_fastapi_app_routes (__main__.TestRunServeIntegration) ... ok
test_uvicorn_app_string_validity (__main__.TestRunServeIntegration) ... ok

----------------------------------------------------------------------
Ran 21 tests in 1.277s
OK
```

### **API Testing with curl**

Test the API endpoints directly:

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_patient.json

# Get API info
curl http://localhost:8000/
```

### **Load Testing**

Test API performance under load:

```bash
# Install Apache Bench (if available)
apt-get install apache2-utils

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -T application/json \
  -p test_patient.json \
  http://localhost:8000/predict
```

## <img src="readme_images/hippo.jpeg" alt="📚" width="20" height="20" style="background: transparent;"> **Understanding the Results**

### **Model Performance Interpretation**

- **AUC > 0.9**: Excellent performance
- **AUC 0.8-0.9**: Good performance, clinically useful
- **AUC 0.7-0.8**: Acceptable performance
- **AUC < 0.7**: Needs improvement

### **Feature Importance Insights**

- **Age**: Strongest predictor (expected)
- **Glucose levels**: Metabolic health indicator
- **Healthcare utilization**: Proxy for overall health status
- **Demographics**: Regional and socioeconomic factors
