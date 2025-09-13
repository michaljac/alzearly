#!/usr/bin/env python3
"""
Windows launcher for Docker Compose services.
Always uses the Windows batch script for startup.
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path


def detect_os():
    """Always return Windows script (assume Windows environment)."""
    return "scripts/start_compose.bat"


def check_script_exists(script_path):
    """Check if the startup script exists."""
    if not Path(script_path).exists():
        raise FileNotFoundError(f"Startup script not found: {script_path}")


def run_script(script_path, args=None):
    """Run the Windows startup script with proper path handling."""
    # Always use Windows path handling
    script_path = script_path.replace('/', '\\')
    cmd = [script_path] + (args or [])
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code: {e.returncode}")
        return e.returncode
    except FileNotFoundError as e:
        print(f"Script not found: {e}")
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Cross-platform Docker Compose launcher for Alzearly"
    )
    parser.add_argument(
        "--retrain", 
        action="store_true", 
        help="Force retraining of the model"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        help="Override data directory path"
    )
    parser.add_argument(
        "--artifacts-dir", 
        type=str, 
        help="Override artifacts directory path"
    )
    
    args = parser.parse_args()
    
    print()
    print("Alzearly Project")
    print("=" * 60)
    print()
    
    try:
        # Always use Windows script
        script_path = detect_os()
        print(f"Using Windows environment")
        print(f"Using script: {script_path}")
        print()
        
        # Check if script exists
        check_script_exists(script_path)
        
        # Prepare environment variables
        env_vars = {}
        if args.retrain:
            env_vars["RETRAIN"] = "1"
        if args.data_dir:
            env_vars["DATA_DIR"] = args.data_dir
        if args.artifacts_dir:
            env_vars["ART_DIR"] = args.artifacts_dir
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")
        
        if env_vars:
            print()
        
        # Run the script
        return_code = run_script(script_path)
        
        if return_code == 0:
            print("Launcher completed successfully!")
        else:
            print(f"Launcher failed with exit code: {return_code}")
        
        return return_code
        
    except Exception as e:
        print(f"Launcher error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
