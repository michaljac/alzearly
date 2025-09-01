#!/usr/bin/env python3
"""
Pipeline script that runs training and then starts the API server.

This script provides a seamless experience from training to serving.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_training_pipeline():
    """Run the complete training pipeline automatically."""
    print("🚀 Starting Alzearly Training Pipeline...")
    print("="*60)
    
    try:
        # Import the automated pipeline function
        from run_training import main
        
        # Run the complete pipeline automatically
        return main() == 0
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        return False

def start_api_server():
    """Start the FastAPI server."""
    print("\n🌐 Starting API Server...")
    print("="*60)
    
    try:
        # Check if models exist
        models_path = Path("artifacts/latest")
        if not models_path.exists():
            print("❌ No trained models found. Please run training first.")
            return False
        
        # Check if FastAPI dependencies are available
        try:
            import fastapi
            import uvicorn
        except ImportError:
            print("❌ FastAPI dependencies not available in training environment")
            print("   To run the API server, use: docker-compose --profile serve up")
            return False
        
        print("🚀 Starting FastAPI server...")
        print("📊 API will be available at: http://localhost:8001")
        print("📖 Interactive docs at: http://localhost:8001/docs")
        print("🛑 Press Ctrl+C to stop the server")
        print()
        
        # Start the server
        import uvicorn
        uvicorn.run(
            "run_serve:app",
            host="0.0.0.0",
            port=8001,
            reload=False,
            log_level="warning"
        )
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ API server failed: {e}")
        return False

def main():
    """Main pipeline function."""
    print("🧠 Alzearly - Complete Pipeline with Optional API Server")
    print("="*60)
    
    # Run training pipeline
    training_success = run_training_pipeline()
    
    if not training_success:
        print("❌ Training failed. Cannot start API server.")
        return 1
    
    print("\n" + "="*60)
    print("🎉 Training Pipeline Completed Successfully!")
    print("="*60)
    
    # Automatically start the API server
    server_success = start_api_server()
    
    if not server_success:
        print("❌ API server failed to start.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
