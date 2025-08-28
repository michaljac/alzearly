@echo off
echo Installing Alzearly Dependencies
echo ================================

echo.
echo Installing PyYAML...
pip install pyyaml==6.0.1

if %ERRORLEVEL% EQU 0 (
    echo ✅ PyYAML installed successfully
) else (
    echo ❌ PyYAML installation failed
    echo Trying alternative installation...
    python -m pip install pyyaml==6.0.1
)

echo.
echo Installing other dependencies...
pip install -r requirements-datagen.txt

if %ERRORLEVEL% EQU 0 (
    echo ✅ Data generation dependencies installed
) else (
    echo ❌ Data generation dependencies failed
)

pip install -r requirements-train.txt

if %ERRORLEVEL% EQU 0 (
    echo ✅ Training dependencies installed
) else (
    echo ❌ Training dependencies failed
)

echo.
echo Testing configuration...
python get_config.py

if %ERRORLEVEL% EQU 0 (
    echo ✅ Configuration system working
    echo.
    echo 🎉 All dependencies installed successfully!
    echo You can now run: train.bat
) else (
    echo ❌ Configuration test failed
    echo The system will use default values
    echo You can still run: train.bat
)

echo.
pause
