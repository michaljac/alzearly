#!/usr/bin/env python3
"""
Simple script to run the Alzheimer's prediction API server.

Usage:
    python run_serve.py [--port PORT] [--host HOST] [--reload]
"""

import argparse
import sys
import uvicorn

def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run Alzheimer's prediction API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("🧠 Alzearly - API Server")
    print("=" * 40)
    print(f"🌐 Server will be available at: http://{args.host}:{args.port}")
    print("📖 Interactive docs at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        uvicorn.run(
            "src.serve:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
