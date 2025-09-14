#!/usr/bin/env python3
"""
Deploy FastAPI application to Google Cloud Run.
Builds Docker image, pushes to Container Registry, and deploys to Cloud Run.
"""

import argparse
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, description=""):
    """Run shell command and handle errors."""
    if description:
        print(f"{description}")
    
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print(f"   {result.stdout.strip()}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ERROR: {e}")
        if e.stderr:
            print(f"   STDERR: {e.stderr}")
        return False, e.stderr


def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    tools = [
        ("docker", ["docker", "--version"]),
        ("gcloud", ["gcloud", "version"]),
    ]
    
    for tool_name, cmd in tools:
        success, output = run_command(cmd, f"Checking {tool_name}")
        if not success:
            print(f"{tool_name} is not installed or not in PATH")
            return False
    
    print("All prerequisites satisfied")
    return True


def configure_docker_auth(project_id):
    """Configure Docker authentication for Google Container Registry."""
    print("\nConfiguring Docker authentication...")
    
    # Configure Docker to use gcloud as credential helper
    success, _ = run_command([
        "gcloud", "auth", "configure-docker", "gcr.io"
    ], "Configuring Docker for GCR")
    
    return success


def build_and_push_image(project_id, service_name, image_tag=None):
    """Build Docker image and push to Google Container Registry."""
    if image_tag is None:
        image_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    image_name = f"gcr.io/{project_id}/{service_name}:{image_tag}"
    latest_image = f"gcr.io/{project_id}/{service_name}:latest"
    
    print(f"\nBuilding and pushing Docker image...")
    print(f"  Image: {image_name}")
    
    # Build image
    build_cmd = [
        "docker", "build", 
        "-t", image_name,
        "-t", latest_image,
        "--platform", "linux/amd64",  # Ensure compatibility with Cloud Run
        "."
    ]
    
    success, _ = run_command(build_cmd, "Building Docker image")
    if not success:
        return False, None
    
    # Push tagged image
    success, _ = run_command([
        "docker", "push", image_name
    ], f"Pushing image {image_name}")
    if not success:
        return False, None
    
    # Push latest tag
    success, _ = run_command([
        "docker", "push", latest_image
    ], f"Pushing latest tag")
    if not success:
        return False, None
    
    print(f"Image pushed successfully: {image_name}")
    return True, image_name


def deploy_to_cloud_run(project_id, service_name, image_name, region, bucket_name, env_vars=None):
    """Deploy image to Cloud Run."""
    print(f"\nDeploying to Cloud Run...")
    print(f"    Region: {region}")
    print(f"    Service: {service_name}")
    
    deploy_cmd = [
        "gcloud", "run", "deploy", service_name,
        "--image", image_name,
        "--platform", "managed",
        "--region", region,
        "--allow-unauthenticated",
        "--port", "8000",
        "--memory", "2Gi",
        "--cpu", "1",
        "--min-instances", "0",
        "--max-instances", "10",
        "--set-env-vars", f"GCS_BUCKET={bucket_name}",
        "--set-env-vars", f"PROJECT_ID={project_id}",
        "--project", project_id
    ]
    
    # Add additional environment variables
    if env_vars:
        for key, value in env_vars.items():
            deploy_cmd.extend(["--set-env-vars", f"{key}={value}"])
    
    success, output = run_command(deploy_cmd, "Deploying to Cloud Run")
    
    if success:
        # Extract service URL from output
        service_url = None
        for line in output.split('\n'):
            if 'https://' in line and region in line:
                service_url = line.strip()
                break
        
        return True, service_url
    
    return False, None


def create_cloud_run_dockerfile():
    """Create optimized Dockerfile for Cloud Run if it doesn't exist."""
    dockerfile_path = Path("Dockerfile.cloudrun")
    
    if dockerfile_path.exists():
        print(f" {dockerfile_path} already exists")
        return str(dockerfile_path)
    
    print(f" Creating {dockerfile_path} for Cloud Run deployment...")
    
    dockerfile_content = '''# Multi-stage Dockerfile for Cloud Run deployment
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port
EXPOSE $PORT

# Start command
CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    print(f" Created {dockerfile_path}")
    return str(dockerfile_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Deploy FastAPI to Google Cloud Run")
    
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="GCP Project ID"
    )
    
    parser.add_argument(
        "--service-name",
        type=str,
        default="alzearly-api",
        help="Cloud Run service name (default: alzearly-api)"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        default="europe-west4",
        help="Cloud Run region (default: europe-west4)"
    )
    
    parser.add_argument(
        "--bucket-name",
        type=str,
        help="GCS bucket name for model artifacts (default: {project-id}-alzearly-models)"
    )
    
    parser.add_argument(
        "--image-tag",
        type=str,
        help="Docker image tag (default: timestamp)"
    )
    
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Docker build and push (use existing image)"
    )
    
    parser.add_argument(
        "--env-vars",
        type=str,
        help="Additional environment variables as JSON string"
    )
    
    args = parser.parse_args()
    
    # Set default bucket name
    if not args.bucket_name:
        args.bucket_name = f"{args.project_id}-alzearly-models"
    
    # Parse additional environment variables
    env_vars = {}
    if args.env_vars:
        try:
            env_vars = json.loads(args.env_vars)
        except json.JSONDecodeError:
            print(" ERROR: Invalid JSON for --env-vars")
            return 1
    
    print(" Alzearly FastAPI Cloud Run Deployment")
    print("=" * 50)
    print(f"  Project ID: {args.project_id}")
    print(f" Service: {args.service_name}")
    print(f" Region: {args.region}")
    print(f" Bucket: {args.bucket_name}")
    if args.image_tag:
        print(f"  Image tag: {args.image_tag}")
    print()
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Step 2: Create Dockerfile if needed
    dockerfile = create_cloud_run_dockerfile()
    
    # Step 3: Configure Docker authentication
    if not configure_docker_auth(args.project_id):
        return 1
    
    # Step 4: Build and push image (unless skipped)
    if not args.skip_build:
        success, image_name = build_and_push_image(args.project_id, args.service_name, args.image_tag)
        if not success:
            return 1
    else:
        image_tag = args.image_tag or "latest"
        image_name = f"gcr.io/{args.project_id}/{args.service_name}:{image_tag}"
        print(f"  Skipping build, using existing image: {image_name}")
    
    # Step 5: Deploy to Cloud Run
    success, service_url = deploy_to_cloud_run(
        args.project_id, 
        args.service_name, 
        image_name, 
        args.region, 
        args.bucket_name,
        env_vars
    )
    
    if success:
        print(f"\n SUCCESS: FastAPI deployed to Cloud Run!")
        if service_url:
            print(f"\n Service URL: {service_url}")
            print(f" Health check: {service_url}/health")
            print(f" API docs: {service_url}/docs")
        
        print(f"\n📋 Manage your service:")
        print(f"   Console: https://console.cloud.google.com/run/detail/{args.region}/{args.service_name}")
        print(f"   Logs: gcloud logging read 'resource.type=\"cloud_run_revision\"' --project={args.project_id}")
        
        return 0
    else:
        print(f"\n FAILED: Deployment unsuccessful")
        return 1


if __name__ == "__main__":
    sys.exit(main())
