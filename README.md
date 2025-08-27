# Alzearly 🧠

> **Early Detection of Alzheimer's Disease**


A comprehensive machine learning pipeline for early detection of Alzheimer's disease using synthetic patient data, featuring XGBoost and Logistic Regression models with automated evaluation and API serving capabilities.

## Quickstart (local, no accounts)

```bash
git clone https://github.com/michaljac/alz-detect
cd alz-detect

# Training pipeline (with interactive tracker selection)
docker build -f Dockerfile.train -t alzearly-train .
docker run -it alzearly-train python run_training.py

# Training with specific tracker
docker run -it alzearly-train python run_training.py --tracker none
docker run -it alzearly-train python run_training.py --tracker wandb
docker run -it alzearly-train python run_training.py --tracker mlflow

# Direct training (bypasses data generation)
docker run -it alzearly-train python src/train.py --tracker none

# Serve latest model (after training)
docker build -f Dockerfile.serve -t alzearly-serve .
docker run -p 8000:8000 alzearly-serve
# Open http://localhost:8000/docs
```

**Notes:**
- Python 3.10 inside all Docker images
- API exposed on port 8000
- Trained artifacts written to `./artifacts/latest/`
- MLflow runs stored in `./mlruns/`
- For Weights & Biases tracking, set `WANDB_API_KEY` environment variable in the docker run command: `-e WANDB_API_KEY=your_key_here`

## <img src="readme_images/hippo.jpeg" alt="🔧" width="25" height="25" style="background: transparent;"> **CLI Commands & Experiment Tracking**

### **Training Commands**

#### **Interactive Tracker Selection (Recommended)**
```bash
# Shows interactive menu to choose tracker
python run_training.py
python src/train.py
```

**User sees:**
```
🔬 Experiment Tracking Setup
==================================================
Select experiment tracker:
1. Weights & Biases (wandb)
2. MLflow (local)
3. No tracking

Enter choice (1-3, default=1): 
```

#### **Non-Interactive Training**
```bash
# No experiment tracking
python src/train.py --tracker none
python run_training.py --tracker none

# With Weights & Biases
export WANDB_API_KEY=your_api_key_here
python src/train.py --tracker wandb
python run_training.py --tracker wandb

# With MLflow (local)
python src/train.py --tracker mlflow
python run_training.py --tracker mlflow
```

### **Key CLI Parameters**

#### **Training Parameters**
```bash
# Basic training
python src/train.py --tracker none

# With custom configuration
python src/train.py \
  --tracker wandb \
  --config config/model.yaml \
  --max-features 100 \
  --handle-imbalance smote \
  --run-type production

# Training with specific data
python src/train.py \
  --tracker none \
  --input-dir data/featurized \
  --output-dir models
```

#### **Available Options**
- `--tracker`: `none`, `wandb`, `mlflow` (or omit for interactive menu)
- `--config`: Configuration file path (default: `config/model.yaml`)
- `--input-dir`: Input data directory (default: `data/featurized`)
- `--output-dir`: Output directory (default: `models`)
- `--max-features`: Maximum features to use (default: 150)
- `--handle-imbalance`: `class_weight`, `smote`, `none` (default: `class_weight`)
- `--run-type`: `initial`, `production`, `hyperparameter_tuned` (default: `initial`)

### **Experiment Tracking Setup**

#### **Weights & Biases (wandb)**
```bash
# Set API key
export WANDB_API_KEY=your_api_key_here

# Run training
python src/train.py --tracker wandb
```

**Features:**
- ✅ Automatic metric logging
- ✅ Model artifact versioning
- ✅ Hyperparameter tracking
- ✅ Visualization dashboards
- ✅ Offline mode support

#### **MLflow (Local)**
```bash
# No setup required
python src/train.py --tracker mlflow
```

**Features:**
- ✅ Local experiment tracking
- ✅ Model registry
- ✅ Metric logging
- ✅ Artifact storage in `./mlruns/`

#### **No Tracking**
```bash
# Fastest option, no external dependencies
python src/train.py --tracker none
```

**Features:**
- ✅ Local artifact saving
- ✅ No external API calls
- ✅ Perfect for development/testing

### **Artifacts & Outputs**

All training runs save artifacts to `artifacts/latest/`:
```bash
artifacts/latest/
├── model.pkl              # Trained model
├── feature_names.json     # Feature names
├── threshold.json         # Optimal threshold
└── metrics.json          # Training metrics
```

### **Docker Commands**

#### **Training with Interactive Menu**
```bash
docker run -it alzearly-train python run_training.py
```

#### **Training with Specific Tracker**
```bash
# No tracking
docker run -it alzearly-train python run_training.py --tracker none

# With wandb
docker run -it -e WANDB_API_KEY=your_key alzearly-train python run_training.py --tracker wandb

# With mlflow
docker run -it alzearly-train python run_training.py --tracker mlflow
```

#### **Direct Training (Skip Data Generation)**
```bash
docker run -it alzearly-train python src/train.py --tracker none
```

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
