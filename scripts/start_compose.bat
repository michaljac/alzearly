@echo off
setlocal enabledelayedexpansion

REM --- Paths & requirements ---
set "DATA_DIR=%DATA_DIR%"
if "%DATA_DIR%"=="" set "DATA_DIR=..\Data\alzearly\featurized"
set "ART_DIR=%ART_DIR%"
if "%ART_DIR%"=="" set "ART_DIR=artifacts\latest"
set "REQ_ART=model.pkl feature_names.json threshold.json"

REM --- Helpers ---
:have_data
if exist "%DATA_DIR%" (
    dir /b "%DATA_DIR%\*" >nul 2>&1
    if not errorlevel 1 (
        goto :have_artifacts
    )
)
goto :no_data

:have_artifacts
for %%f in (%REQ_ART%) do (
    if not exist "%ART_DIR%\%%f" (
        goto :no_artifacts
    )
)
echo ✅ Found all required artifacts - skipping training
goto :start_server

:no_data
echo 📦 No featurized data found — generating...
docker compose run --rm datagen
goto :check_retrain

:no_artifacts
echo 🏋️ No artifacts — training once...
docker compose run --rm training
goto :start_server

:check_retrain
if "%RETRAIN%"=="1" (
    echo ♻️ RETRAIN=1 — running training now...
    docker compose run --rm training
) else (
    echo ✅ Artifacts already present — skip training.
)

:start_server
echo 🚀 Starting API server...

REM Read configuration from serve.yaml
for /f "tokens=2 delims=: " %%a in ('findstr "app_host:" config\serve.yaml') do set "APP_HOST=%%a"
for /f "tokens=2 delims=: " %%a in ('findstr "app_port:" config\serve.yaml') do set "APP_PORT=%%a"

REM Remove quotes if present
set "APP_HOST=%APP_HOST:"=%"
set "APP_PORT=%APP_PORT:"=%"

REM Set default values if not found
if "%APP_HOST%"=="" set "APP_HOST=0.0.0.0"
if "%APP_PORT%"=="" set "APP_PORT=8001"

echo 🌐 Using host: %APP_HOST%
echo 🔌 Using port: %APP_PORT%

REM Start the serve service
docker compose up -d serve

echo 📋 Current services:
docker compose ps

echo ✅ Done. Open the docs at http://localhost:%APP_PORT%/docs

endlocal
