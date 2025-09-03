@echo off
setlocal enabledelayedexpansion

REM --- Paths & requirements ---
set "DATA_DIR=%DATA_DIR%"
if "%DATA_DIR%"=="" set "DATA_DIR=..\Data\alzearly\featurized"
set "ART_DIR=%ART_DIR%"
if "%ART_DIR%"=="" set "ART_DIR=artifacts\latest"
set "REQ_ART=model.pkl feature_names.json run_log.json"

REM --- Helpers ---
:check_data
echo Checking data...
set "HAS_DATA=0"
if exist "%DATA_DIR%" (
    for %%f in ("%DATA_DIR%\*") do (
        if not "%%~nxf"=="" (
            set "HAS_DATA=1"
        )
    )
)
if "!HAS_DATA!"=="1" (
    echo Data found
    echo.
    goto :check_artifacts
)
echo Generating data...
if not exist "..\Data\alzearly\raw" md "..\Data\alzearly\raw" >nul
if not exist "..\Data\alzearly\featurized" md "..\Data\alzearly\featurized" >nul
docker compose run --rm datagen >nul
if errorlevel 1 (
    echo Data generation failed
    exit /b 1
)
echo Data generated

:check_artifacts
echo Checking model...
for %%f in (%REQ_ART%) do (
    if not exist "%ART_DIR%\%%f" (
        echo Training model...
        docker compose run --rm training >nul
        if errorlevel 1 (
            echo Training failed
            exit /b 1
        )
        echo Model trained
        goto :start_server
    )
)
echo Model ready
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

echo Access: http://localhost:%APP_PORT%

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
