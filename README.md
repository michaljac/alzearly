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

```bash
# Clone the repository
git clone <repository-url>
cd alzheimers-prediction

# Install dependencies
pip install -r requirements.txt
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
├── requirements.txt           # Dependencies
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
- [ ] **Model serving**: REST API for real-time predictions
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.

---

**Note**: This project uses synthetic data for demonstration purposes. For real clinical applications, ensure compliance with relevant healthcare data regulations and privacy requirements.
