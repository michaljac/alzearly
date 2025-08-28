# PowerShell script for Alzearly Training Pipeline
Write-Host "Starting Alzearly Training Pipeline" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Get the current directory and Data directory (platform-independent)
$CURRENT_DIR = Get-Location
$DATA_DIR = (Resolve-Path "../Data/alzearly").Path

# Create Data directory if it doesn't exist
if (!(Test-Path $DATA_DIR)) {
    New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null
}

Write-Host "Checking for existing data in $DATA_DIR/featurized..." -ForegroundColor Yellow

# Check if featurized data exists
$FEATURIZED_DIR = Join-Path $DATA_DIR "featurized"
$DATA_FOUND = $false

if (Test-Path $FEATURIZED_DIR) {
    $PARQUET_FILES = Get-ChildItem -Path $FEATURIZED_DIR -Filter "*.parquet" -ErrorAction SilentlyContinue
    $CSV_FILES = Get-ChildItem -Path $FEATURIZED_DIR -Filter "*.csv" -ErrorAction SilentlyContinue
    
    if ($PARQUET_FILES -or $CSV_FILES) {
        $DATA_FOUND = $true
    }
}

if (-not $DATA_FOUND) {
    Write-Host "No featurized data found" -ForegroundColor Red
    Write-Host "Generating data using datagen container..." -ForegroundColor Yellow
    
    # Run data generation container
    docker run --rm `
        -v "${CURRENT_DIR}:/workspace" `
        -v "${DATA_DIR}:/Data" `
        alzearly-datagen:latest `
        python run_datagen.py
    
    Write-Host "Data generation completed" -ForegroundColor Green
} else {
    Write-Host "Found existing featurized data" -ForegroundColor Green
}

Write-Host "Starting training..." -ForegroundColor Yellow

# Run training container
docker run --rm `
    -v "${CURRENT_DIR}:/workspace" `
    -v "${DATA_DIR}:/Data" `
    alzearly-train:latest `
    python run_training.py

Write-Host "Training pipeline completed!" -ForegroundColor Green
