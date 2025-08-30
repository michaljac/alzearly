@echo off
setlocal enabledelayedexpansion

REM Simple color scheme for Windows compatibility
set "RED="
set "GREEN="
set "YELLOW="
set "BLUE="
set "PURPLE="
set "CYAN="
set "WHITE="
set "NC="

REM Parse command line arguments
set FORCE_REGEN=false
set SKIP_DATA_GEN=false
set SKIP_PREPROCESS=true
set START_SERVER=false
set SERVER_PORT=
set SERVER_HOST=0.0.0.0
set TRACKER=

:parse_args
if "%1"=="" goto :end_parse
if "%1"=="--force-regen" (
    set FORCE_REGEN=true
    shift
    goto :parse_args
)
if "%1"=="--skip-data-gen" (
    set SKIP_DATA_GEN=true
    shift
    goto :parse_args
)
if "%1"=="--skip-preprocess" (
    set SKIP_PREPROCESS=true
    shift
    goto :parse_args
)
if "%1"=="--serve" (
    set START_SERVER=true
    shift
    goto :parse_args
)
if "%1"=="--port" (
    set SERVER_PORT=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--host" (
    set SERVER_HOST=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--tracker" (
    set TRACKER=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--help" (
    echo %CYAN%Alzearly Training Pipeline%NC%
    echo Usage: train.bat [options]
    echo.
    echo Options:
    echo   %GREEN%--force-regen%NC%           Force regeneration of data
    echo   %GREEN%--skip-data-gen%NC%         Skip data generation step
    echo   %GREEN%--skip-preprocess%NC%       Skip preprocessing step ^(default: true^)
    echo   %GREEN%--serve%NC%                 Start API server after training
    echo   %GREEN%--port PORT%NC%             Specify port for server ^(auto-find if not specified^)
    echo   %GREEN%--host HOST%NC%             Specify host for server ^(default: 0.0.0.0^)
    echo   %GREEN%--tracker TRACKER%NC%       Specify experiment tracker ^(none, wandb, mlflow^) ^(interactive prompt if not specified^)
    echo   %GREEN%--help%NC%                  Show this help message
    echo.
    echo Examples:
    echo   %YELLOW%train.bat%NC%                           # Train with interactive tracker selection
    echo   %YELLOW%train.bat --tracker none%NC%            # Train without tracking
    echo   %YELLOW%train.bat --tracker wandb%NC%           # Train with Weights & Biases tracking
    echo   %YELLOW%train.bat --tracker mlflow%NC%          # Train with MLflow tracking
    echo   %YELLOW%train.bat --serve%NC%                   # Train with interactive tracker + start server
    echo   %YELLOW%train.bat --serve --port 8001%NC%       # Train with interactive tracker + start server on port 8001
    echo   %YELLOW%train.bat --force-regen%NC%             # Force regenerate data
    echo.
    echo Configuration is read from config/data_gen.yaml
    exit /b 0
)
shift
goto :parse_args
:end_parse

REM Read configuration from config/data_gen.yaml
echo Reading configuration from config/data_gen.yaml...
for /f "tokens=1,2 delims==" %%a in ('python get_config.py 2^>nul') do (
    set %%a=%%b
)

REM Set default values if config reading failed
if not defined CONFIG_N_PATIENTS set CONFIG_N_PATIENTS=3000
if not defined CONFIG_YEARS set CONFIG_YEARS=2021,2022,2023,2024
if not defined CONFIG_POSITIVE_RATE set CONFIG_POSITIVE_RATE=0.08
if not defined CONFIG_SEED set CONFIG_SEED=42
if not defined CONFIG_OUTPUT_DIR set CONFIG_OUTPUT_DIR=data/raw

echo.
echo %PURPLE%🧠 Alzearly Training Pipeline%NC%
echo %CYAN%========================================%NC%
echo.
echo %WHITE%Configuration from config/data_gen.yaml:%NC%
echo   %BLUE%Number of patients:%NC% %CYAN%%CONFIG_N_PATIENTS%%NC%
if not "%TRACKER%"=="" (
    echo   %BLUE%Experiment tracker:%NC% %CYAN%%TRACKER%%NC%
) else (
    echo   %BLUE%Experiment tracker:%NC% %CYAN%Interactive prompt%NC%
    echo   %YELLOW%💡 You will be prompted to choose a tracker during training%NC%
)
echo   %BLUE%Years:%NC% %CYAN%%CONFIG_YEARS%%NC%
echo   %BLUE%Positive rate:%NC% %CYAN%%CONFIG_POSITIVE_RATE%%NC%
echo   %BLUE%Seed:%NC% %CYAN%%CONFIG_SEED%%NC%
echo   %BLUE%Output directory:%NC% %CYAN%%CONFIG_OUTPUT_DIR%%NC%
echo   %BLUE%Force regeneration:%NC% %CYAN%%FORCE_REGEN%%NC%
echo   %BLUE%Skip data generation:%NC% %CYAN%%SKIP_DATA_GEN%%NC%
echo   %BLUE%Skip preprocessing:%NC% %CYAN%%SKIP_PREPROCESS%%NC%
echo   %BLUE%Start server:%NC% %CYAN%%START_SERVER%%NC%
if not "%SERVER_PORT%"=="" (
    echo   %BLUE%Server port:%NC% %CYAN%%SERVER_PORT%%NC%
) else (
    echo   %BLUE%Server port:%NC% %CYAN%auto-find%NC%
)
echo.

REM Get the current directory and Data directory (platform-independent)
set CURRENT_DIR=%cd%
for %%i in ("%cd%\..\Data\alzearly") do set DATA_DIR=%%~fi

REM Create Data directory if it doesn't exist
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

echo %BLUE%📁%NC% Checking for existing data in %CYAN%%DATA_DIR%/featurized%NC%...

REM Check if featurized data exists
set DATA_FOUND=false
if exist "%DATA_DIR%\featurized\*.parquet" set DATA_FOUND=true
if exist "%DATA_DIR%\featurized\*.csv" set DATA_FOUND=true

if "%DATA_FOUND%"=="false" (
    echo %RED%❌%NC% No featurized data found
    if "%SKIP_DATA_GEN%"=="false" (
        echo %YELLOW%🔄%NC% Generating data using datagen container...
        echo.
        
        REM Run data generation container with config parameters and Polars fixes
        docker run --rm -v "%CURRENT_DIR%:/workspace" -v "%DATA_DIR%:/Data" -e PYTHONUNBUFFERED=1 -e TQDM_DISABLE=1 -e POLARS_NUM_THREADS=1 -e POLARS_STREAMING_COLLECT=0 -e POLARS_FORCE_OPTIMIZED=1 -e POLARS_USE_STREAMING=0 alzearly-datagen:latest python run_datagen.py --output-dir /Data/raw --num-patients %CONFIG_N_PATIENTS% --seed %CONFIG_SEED%
        
        echo %GREEN%✅%NC% Data generation completed
        echo.
        
        REM Preprocessing is now handled automatically by run_datagen.py
        echo %BLUE%🔧%NC% Preprocessing completed automatically
        echo %GREEN%✅%NC% Preprocessing completed
        echo.
    ) else (
        echo %YELLOW%⏭️%NC% Data generation skipped
        echo.
    )
) else (
    if "%FORCE_REGEN%"=="true" (
        echo %YELLOW%🔄%NC% Force regenerating data...
        docker run --rm -v "%CURRENT_DIR%:/workspace" -v "%DATA_DIR%:/Data" -e PYTHONUNBUFFERED=1 -e TQDM_DISABLE=1 -e POLARS_NUM_THREADS=1 -e POLARS_STREAMING_COLLECT=0 -e POLARS_FORCE_OPTIMIZED=1 -e POLARS_USE_STREAMING=0 alzearly-datagen:latest python run_datagen.py --output-dir /Data/raw --num-patients %CONFIG_N_PATIENTS% --seed %CONFIG_SEED% --force-regen
        echo %GREEN%✅%NC% Data generation completed
        echo.
    ) else (
        echo %GREEN%✅%NC% Found existing featurized data
        echo.
    )
)

echo.
echo %PURPLE%🤖%NC% Starting training...
echo %YELLOW%⏳%NC% This may take a few minutes...
echo %BLUE%ℹ️%NC% Using optimized settings for stability...
echo.
REM Run training container with interactive mode for tracker selection
if not "%TRACKER%"=="" (
    docker run --rm -v "%CURRENT_DIR%:/workspace" -v "%DATA_DIR%:/Data" -e PYTHONUNBUFFERED=1 -e TQDM_DISABLE=1 -e POLARS_NUM_THREADS=1 -e POLARS_STREAMING_COLLECT=0 -e POLARS_FORCE_OPTIMIZED=1 alzearly-train:latest python run_training.py --tracker %TRACKER% --config config/model.yaml
) else (
    docker run -it --rm -v "%CURRENT_DIR%:/workspace" -v "%DATA_DIR%:/Data" -e PYTHONUNBUFFERED=1 -e TQDM_DISABLE=1 -e POLARS_NUM_THREADS=1 -e POLARS_STREAMING_COLLECT=0 -e POLARS_FORCE_OPTIMIZED=1 alzearly-train:latest python run_training.py --config config/model.yaml
)

echo.
echo %GREEN%🎉%NC% Training pipeline completed!
echo.

REM Automatically start the API server after successful training
echo %CYAN%🚀 Starting API Server automatically after training%NC%
echo %CYAN%==================================================%NC%
echo.
echo %BLUE%🌐%NC% Starting API server...
echo %BLUE%📖%NC% Interactive docs will be available at: %CYAN%http://localhost:8001/docs%NC%
echo %YELLOW%🛑%NC% Press Ctrl+C to stop the server
echo.

REM Run the API server with auto port detection using Docker container
docker run --rm -v "%CURRENT_DIR%:/workspace" -p 8001:8001 -w /workspace alzearly-serve:latest python run_serve.py --port 8001

REM Legacy --serve option (now redundant since server starts automatically)
if "%START_SERVER%"=="true" (
    echo.
    echo %YELLOW%⚠️%NC% Server is already running automatically. The --serve option is no longer needed.
    echo.
)

pause
