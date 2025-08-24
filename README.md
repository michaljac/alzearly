# Alzheimer's Disease Prediction Project

A comprehensive machine learning pipeline for predicting Alzheimer's disease using synthetic clinical data with realistic features and temporal patterns.

## 🎯 Project Overview

This project implements a complete ML pipeline for Alzheimer's disease prediction, featuring:

- **Synthetic Data Generation**: Realistic patient-year clinical data with 5-10% positive prevalence
- **Feature Engineering**: 191 features including rolling statistics, temporal changes, and risk factors
- **Memory-Efficient Processing**: Polars Lazy evaluation for large-scale datasets
- **Configuration-Driven**: YAML-based configuration for all pipeline components
- **Scalable Architecture**: Designed to handle millions of rows with chunked processing

## 📊 Dataset Characteristics

### Data Structure
- **Format**: Patient-year (one row per patient per calendar year)
- **Scale**: 2M+ rows (configurable)
- **Features**: 27 base + 164 engineered = 191 total features
- **Target**: Alzheimer's diagnosis (5-10% positive prevalence)
- **Storage**: Partitioned Parquet files by year

### Feature Categories
- **Demographics**: Sex, region, occupation, education, marital status, insurance
- **Clinical**: Age, BMI, blood pressure, glucose, cholesterol, lab values
- **Temporal**: Rolling statistics, year-over-year changes, patient aggregates
- **Risk Factors**: BMI categories, BP categories, age groups, composite risk scores

## 🚀 Quick Start

### 1. Installation

**Choose the appropriate requirements file based on your needs:**
- `requirements-train.txt` - Full development environment (data generation, preprocessing, training)
- `requirements-serve.txt` - Lightweight serving environment (model inference only)

```bash
# Clone the repository
git clone <repository-url>
cd alzheimers-prediction

# Install dependencies for training (includes all packages)
pip install -r requirements-train.txt

# Or install dependencies for serving only
pip install -r requirements-serve.txt
```

### 2. Generate Data

```bash
# Generate data using default configuration
python cli.py data-gen

# Generate with custom parameters
python cli.py data-gen --config config/data_gen.yaml --n-patients 50000 --positive-rate 0.08

# Generate 2M rows (500K patients × 4 years)
python cli.py data-gen --n-patients 500000
```

### 3. Preprocess Data

```bash
# Preprocess with default configuration
python cli.py preprocess

# Preprocess with custom configuration
python cli.py preprocess --config config/preprocess.yaml
```

### 4. Train Model

```bash
# Train model with default configuration
python cli.py train

# Train with custom configuration
python cli.py train --config config/model.yaml
```

### 5. Serve Model

```bash
# Start the FastAPI server
uvicorn src.serve:app --host 0.0.0.0 --port 8002

# The API will be available at:
# - API Documentation: http://localhost:8002/docs
# - Health Check: http://localhost:8002/health
# - Prediction Endpoint: http://localhost:8002/predict
```

**Test the API:**
```bash
# Run the test script to verify all endpoints
python test_api.py
```

## 🌐 API Documentation

### FastAPI Server

The project includes a production-ready FastAPI server for real-time Alzheimer's disease predictions.

#### Endpoints

- **`GET /`** - API information and available endpoints
- **`GET /health`** - Health check with model status
- **`POST /predict`** - Make predictions for patient data
- **`GET /docs`** - Interactive API documentation (Swagger UI)

#### Example Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8002/health")
print(response.json())

# Make prediction
patient_data = {
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

response = requests.post("http://localhost:8002/predict", json=patient_data)
prediction = response.json()
print(f"Probability: {prediction['probability']:.3f}")
print(f"Prediction: {'High Risk' if prediction['label'] == 1 else 'Low Risk'}")
```

## ⚙️ Configuration System

The project uses YAML-based configuration files for all components:

### Data Generation (`config/data_gen.yaml`)

```yaml
# Dataset size and structure
dataset:
  n_patients: 100000
  years: [2018, 2019, 2020, 2021]
  target_rows: null  # Overrides n_patients if specified

# Target variable configuration
target:
  positive_rate: 0.08  # 5-10% recommended
  column_name: "alzheimers_diagnosis"

# Processing configuration
processing:
  rows_per_chunk: 100000
  seed: 42

# Clinical feature ranges
clinical_ranges:
  age:
    min: 18
    max: 85
  bmi:
    min: 16.0
    max: 50.0
  # ... more ranges
```

### Preprocessing (`config/preprocess.yaml`)

```yaml
# Input/Output configuration
io:
  input_dir: "data/raw"
  output_dir: "data/featurized"

# Feature engineering
features:
  target_column: "alzheimers_diagnosis"
  numeric_columns:
    - "age"
    - "bmi"
    - "systolic_bp"
    # ... more columns

# Rolling features
rolling:
  window_years: 3
  statistics: ["mean", "max", "sum", "count"]

# Risk features
risk_features:
  bmi_categories:
    underweight_threshold: 18.5
    normal_max: 25.0
    overweight_max: 30.0
  # ... more thresholds
```

### Model Training (`config/model.yaml`)

```yaml
# Model configuration
model:
  type: "xgboost"
  name: "alzheimers_predictor"

# Training configuration
training:
  test_size: 0.2
  validation_size: 0.2
  cv_folds: 5
  stratified: true

# Feature selection
feature_selection:
  enabled: true
  method: "mutual_info"
  n_features: 50

# Hyperparameter tuning
hyperparameter_tuning:
  enabled: true
  method: "bayesian"
  n_trials: 100
```

## 📁 Project Structure

```
alzheimers-prediction/
├── config/                     # Configuration files
│   ├── data_gen.yaml          # Data generation config
│   ├── preprocess.yaml        # Preprocessing config
│   └── model.yaml             # Model training config
├── src/                       # Source code
│   ├── config.py              # Configuration loader
│   ├── data_gen.py            # Data generation
│   ├── preprocess.py          # Feature engineering
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   └── serve.py               # Model serving
├── data/                      # Data directories
│   ├── raw/                   # Raw generated data
│   └── featurized/            # Processed features
├── models/                    # Trained models
├── results/                   # Training results
├── logs/                      # Log files
├── cli.py                     # Command-line interface
├── requirements-train.txt     # Training dependencies
├── requirements-serve.txt     # Serving dependencies
└── README.md                  # This file
```

## 🔧 Command Line Interface

### Data Generation

```bash
# Basic usage
python cli.py data-gen

# With configuration file
python cli.py data-gen --config config/data_gen.yaml

# Override config parameters
python cli.py data-gen --n-patients 50000 --positive-rate 0.08

# Generate specific number of rows
python cli.py data-gen --config config/data_gen.yaml
# Edit config/data_gen.yaml to set target_rows: 2000000
```

### Preprocessing

```bash
# Basic usage
python cli.py preprocess

# With custom configuration
python cli.py preprocess --config config/preprocess.yaml

# Override parameters
python cli.py preprocess --input-dir data/raw --output-dir data/featurized
```

### Model Training

```bash
# Basic usage
python cli.py train

# With custom configuration
python cli.py train --config config/model.yaml

# Override parameters
python cli.py train --model-type xgboost --n-trials 200
```

## 📈 Performance Characteristics

### Memory Usage
- **2M rows**: ~2.8 GB
- **5M rows**: ~7.1 GB  
- **10M rows**: ~14.2 GB

### Feature Count
- **Base features**: 27 (demographics, clinical, encounters)
- **Rolling features**: 60 (4 per numeric column)
- **Delta features**: 30 (2 per numeric column)
- **Aggregate features**: 60 (4 per numeric column)
- **Risk features**: 14 (categories and scores)
- **Total**: 191 features

### Processing Speed
- **Data generation**: ~100K rows/minute
- **Feature engineering**: ~50K rows/minute
- **Model training**: Depends on feature selection and hyperparameter tuning

## 🎯 Key Features

### Realistic Data Generation
- **Temporal consistency**: Patient states maintained across years
- **Clinical correlations**: Health metrics influence encounter patterns
- **Risk-based prevalence**: Alzheimer's probability based on medical risk factors
- **Age progression**: Realistic aging patterns and health deterioration

### Efficient Processing
- **Lazy evaluation**: Polars LazyFrame for memory efficiency
- **Chunked processing**: Configurable chunk sizes for large datasets
- **Partitioned storage**: Year-based partitioning for efficient querying
- **Columnar format**: Parquet compression for storage efficiency

### Comprehensive Feature Engineering
- **Rolling statistics**: Mean, max, sum, count over time windows
- **Temporal changes**: Year-over-year deltas and percentage changes
- **Patient aggregates**: Lifetime statistics per patient
- **Risk categorization**: Medical risk factors and composite scores

### Configuration-Driven
- **YAML configuration**: Human-readable parameter management
- **Override support**: Command-line overrides for config values
- **Validation**: Automatic configuration validation and error handling
- **Modular design**: Separate configs for each pipeline stage

## 🎯 Threshold Selection Strategy

The model evaluation uses a **two-tier threshold selection** approach for optimal binary classification:

### Primary Strategy: Maximum F1 Score
- **Goal**: Find threshold that maximizes F1-score (harmonic mean of precision and recall)
- **Why F1?** Balances false positives (unnecessary alarms) and false negatives (missed cases)
- **Medical Context**: Critical for Alzheimer's diagnosis where both precision and recall matter

### Fallback Strategy: Recall Target
- **Goal**: Ensure minimum 80% recall (catch at least 80% of actual cases)
- **Why Fallback?** Sometimes maximizing F1 results in too low recall
- **Medical Priority**: Missing actual cases can be more dangerous than false alarms

### Why This Approach?
For Alzheimer's prediction, I chose this strategy because:
- **F1 optimization**: Provides balanced performance for general screening
- **80% recall fallback**: Ensures we don't miss too many actual cases in critical scenarios
- **Medical safety**: Prioritizes catching real cases over avoiding false alarms
- **Flexibility**: Two thresholds allow different use cases (screening vs. high-risk monitoring)

### Implementation
```python
# Test thresholds from 0.1 to 0.9
thresholds = [0.1, 0.15, 0.2, ..., 0.85, 0.9]

# Choose optimal threshold
optimal_threshold = threshold_with_max_f1_score
fallback_threshold = threshold_with_recall >= 0.8
```

**Result**: Two thresholds saved to `threshold.json` - use optimal for general cases, fallback when high recall is critical.

## 🔬 Technical Details

### Data Generation Algorithm
1. **Patient initialization**: Generate stable demographic and baseline clinical features
2. **Temporal progression**: Apply random walks to simulate health changes over time
3. **Risk calculation**: Compute Alzheimer's probability based on age and risk factors
4. **Diagnosis generation**: Sample from probability distribution to create target variable
5. **Chunked writing**: Write data in configurable chunks to partitioned Parquet files

### Feature Engineering Pipeline
1. **Data loading**: Read partitioned Parquet files using Polars Lazy
2. **Rolling features**: Compute statistics over configurable time windows
3. **Delta features**: Calculate year-over-year changes and percentage changes
4. **Aggregate features**: Compute patient-level statistics across all years
5. **Risk features**: Create categorical features and composite risk scores
6. **Missing value handling**: Fill null values with appropriate defaults

### Memory Optimization
- **Lazy evaluation**: Defer computation until materialization
- **Chunked processing**: Process data in configurable chunks
- **Efficient data types**: Use appropriate data types for memory efficiency
- **Streaming operations**: Avoid loading entire dataset into memory

## 🚧 Future Enhancements

### Planned Features
- [x] **Model serving**: REST API for real-time predictions ✅
- [ ] **Feature selection**: Automated feature importance analysis
- [ ] **Hyperparameter optimization**: Advanced tuning strategies
- [ ] **Model interpretability**: SHAP explanations and feature importance
- [ ] **Data validation**: Schema validation and data quality checks
- [ ] **Monitoring**: Model performance monitoring and drift detection

### Scalability Improvements
- [ ] **Distributed processing**: Spark integration for very large datasets
- [ ] **Cloud deployment**: AWS/GCP deployment configurations
- [ ] **Caching**: Redis caching for frequently accessed data
- [ ] **Parallel processing**: Multi-core feature engineering


**Note**: This project uses synthetic data for demonstration purposes. For real clinical applications, ensure compliance with relevant healthcare data regulations and privacy requirements.
