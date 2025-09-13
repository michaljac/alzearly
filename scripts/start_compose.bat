@echo off
setlocal enabledelayedexpansion

REM --- Paths & requirements ---
set "DATA_DIR=%DATA_DIR%"
if "%DATA_DIR%"=="" set "DATA_DIR=..\Data\alzearly\featurized"
set "RAW_DIR=..\Data\alzearly\raw"
set "ART_DIR=%ART_DIR%"
if "%ART_DIR%"=="" set "ART_DIR=artifacts\latest"
set "REQ_ART=model.pkl feature_names.json run_log.json"

REM --- Helpers ---
:check_data
echo Checking data...
set "HAS_RAW_DATA=0"
set "HAS_FEATURIZED_DATA=0"

REM Check for raw data (year directories with actual files)

if exist "%RAW_DIR%" (
    dir "%RAW_DIR%\*\*.parquet" >nul 2>&1
    if not errorlevel 1 (
        set "HAS_RAW_DATA=1"
    ) else (
        dir "%RAW_DIR%\*\*" >nul 2>&1
        if not errorlevel 1 (
            set "HAS_RAW_DATA=1"
        )
    )
)

REM Check for featurized data (actual parquet files)

if exist "%DATA_DIR%\*.parquet" (
    set "HAS_FEATURIZED_DATA=1"
)

if "!HAS_RAW_DATA!"=="1" (
    echo Data found. Regenerate? (y/n^)
    set /p REGEN_DATA=
    if /i "!REGEN_DATA!"=="y" (
        goto :generate_data
    ) else (
        echo Using existing data
        REM Still check if we need featurized data
        if "!HAS_FEATURIZED_DATA!"=="0" (
            echo No featurized data found. Preprocessing required.
        )
        echo.
        goto :check_artifacts
    )
) else if "!HAS_FEATURIZED_DATA!"=="1" (
    echo Data found. Regenerate? (y/n^)
    set /p REGEN_DATA=
    if /i "!REGEN_DATA!"=="y" (
        goto :generate_data
    ) else (
        echo Using existing featurized data
        echo.
        goto :check_artifacts
    )
) else (
    echo No existing data found. Generating fresh data...
)
:generate_data
echo Generating data...
if not exist "..\Data\alzearly\raw" md "..\Data\alzearly\raw" >nul
if not exist "..\Data\alzearly\featurized" md "..\Data\alzearly\featurized" >nul

docker compose run --rm datagen
if errorlevel 1 (
    echo Data generation failed
    exit /b 1
)

docker compose run --rm preprocess
if errorlevel 1 (
    echo Preprocessing failed
    exit /b 1
)

:check_artifacts
echo Checking model...
set "HAS_MODEL=1"
for %%f in (%REQ_ART%) do (
    if not exist "%ART_DIR%\%%f" (
        set "HAS_MODEL=0"
    )
)
if "!HAS_MODEL!"=="1" (
    echo Model found. Retrain? (y/n^)
    set /p RETRAIN_MODEL=
    if /i "!RETRAIN_MODEL!"=="y" (
        echo Retraining model...
        goto :train_model
    ) else (
        echo Using existing model
        echo.
        goto :start_server
    )
)
echo No model found. Training model...
:train_model
docker compose run --rm training >nul
if errorlevel 1 (
    echo Training failed
    exit /b 1
)
echo Model trained
echo.

:check_retrain
if "%RETRAIN%"=="1" (
    echo Retraining...
    docker compose run --rm training >nul
    if errorlevel 1 (
        echo Retraining failed
        exit /b 1
    )
    echo Retraining done
)

:start_server
echo Starting server...

REM Read configuration
for /f "tokens=2 delims=: " %%a in ('findstr "app_host:" config\serve.yaml') do set "APP_HOST=%%a"
for /f "tokens=2 delims=: " %%a in ('findstr "app_port:" config\serve.yaml') do set "APP_PORT=%%a"

REM Set defaults
if "%APP_HOST%"=="" set "APP_HOST=0.0.0.0"
if "%APP_PORT%"=="" set "APP_PORT=8001"

@REM echo Access: http://localhost:%APP_PORT%

REM Start service silently
docker compose up -d serve >nul

REM Simple status check
timeout /t 5 /nobreak >nul
docker compose ps serve | findstr "Up" >nul
if errorlevel 1 (
    echo Service failed
) else (
    echo Service running
)
echo.

echo Ready: http://localhost:%APP_PORT%/docs

endlocal

endlocal
