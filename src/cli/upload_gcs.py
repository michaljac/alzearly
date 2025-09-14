#!/usr/bin/env python3
"""
Upload model artifacts to Google Cloud Storage.
Checks for existing artifacts and uploads them to GCS bucket for Cloud Run deployment.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from google.cloud.exceptions import NotFound


def check_artifacts(artifacts_dir):
    """Check if all required artifacts exist."""
    required_artifacts = [
        "model.pkl",
        "feature_names.json", 
        "run_log.json"
    ]
    
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        print(f"Artifacts directory {artifacts_dir} does not exist")
        return False, []
    
    missing_artifacts = []
    existing_artifacts = []
    
    for artifact in required_artifacts:
        artifact_path = artifacts_path / artifact
        if artifact_path.exists():
            existing_artifacts.append(artifact_path)
            print(f"Found: {artifact}")
        else:
            missing_artifacts.append(artifact)
            print(f"Missing: {artifact}")
    
    if missing_artifacts:
        print(f"\nMissing artifacts: {missing_artifacts}")
        return False, existing_artifacts
    
    print(f"\nAll required artifacts found in {artifacts_dir}")
    return True, existing_artifacts


def create_storage_client():
    """Create Google Cloud Storage client with authentication."""
    try:
        client = storage.Client()
        return client
    except Exception as e:
        print(f"ERROR: Failed to create Storage client: {e}")
        print("Make sure you're authenticated with: gcloud auth application-default login")
        print("Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return None


def create_bucket_if_not_exists(client, bucket_name, project_id, location="europe-west4"):
    """Create bucket if it doesn't exist."""
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists")
        return bucket
    except NotFound:
        print(f"Creating bucket {bucket_name}...")
        try:
            bucket = client.create_bucket(bucket_name, project=project_id, location=location)
            print(f"Created bucket {bucket_name} in {location}")
            return bucket
        except Exception as e:
            print(f"ERROR: Failed to create bucket: {e}")
            return None


def upload_artifacts(client, bucket_name, artifacts_dir, artifacts_list, model_version=None):
    """Upload artifacts to GCS bucket."""
    bucket = client.bucket(bucket_name)
    
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    uploaded_files = []
    
    print(f"\nUploading artifacts to gs://{bucket_name}/models/{model_version}/")
    
    for artifact_path in artifacts_list:
        try:
            # Define GCS path
            gcs_path = f"models/{model_version}/{artifact_path.name}"
            blob = bucket.blob(gcs_path)
            
            # Upload file
            blob.upload_from_filename(str(artifact_path))
            uploaded_files.append(f"gs://{bucket_name}/{gcs_path}")
            
            print(f"Uploaded: {artifact_path.name} → gs://{bucket_name}/{gcs_path}")
            
        except Exception as e:
            print(f"ERROR uploading {artifact_path.name}: {e}")
            return False, uploaded_files
    
    # Upload latest symlink info
    try:
        latest_info = {
            "model_version": model_version,
            "uploaded_at": datetime.now().isoformat(),
            "artifacts": [f.split('/')[-1] for f in uploaded_files]
        }
        
        import json
        latest_blob = bucket.blob("models/latest.json")
        latest_blob.upload_from_string(json.dumps(latest_info, indent=2))
        
        print(f"Updated latest model pointer: gs://{bucket_name}/models/latest.json")
        
    except Exception as e:
        print(f"WARNING: Failed to update latest pointer: {e}")
    
    return True, uploaded_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Upload model artifacts to Google Cloud Storage")
    
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/latest",
        help="Directory containing model artifacts (default: artifacts/latest)"
    )
    
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="GCP Project ID"
    )
    
    parser.add_argument(
        "--bucket-name",
        type=str,
        help="GCS bucket name (default: {project-id}-alzearly-models)"
    )
    
    parser.add_argument(
        "--model-version",
        type=str,
        help="Model version tag (default: timestamp)"
    )
    
    parser.add_argument(
        "--location",
        type=str,
        default="europe-west4",
        help="GCS bucket location (default: europe-west4)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if some artifacts are missing"
    )
    
    args = parser.parse_args()
    
    # Set default bucket name
    if not args.bucket_name:
        args.bucket_name = f"{args.project_id}-alzearly-models"
    
    print("Alzearly Model Artifact Upload to GCS")
    print("=" * 50)
    print(f"Artifacts directory: {args.artifacts_dir}")
    print(f"Project ID: {args.project_id}")
    print(f"Bucket: {args.bucket_name}")
    print(f"Location: {args.location}")
    if args.model_version:
        print(f"🏷️  Version: {args.model_version}")
    print()
    
    # Step 1: Check artifacts
    artifacts_exist, artifacts_list = check_artifacts(args.artifacts_dir)
    
    if not artifacts_exist and not args.force:
        print("\nNot all required artifacts exist. Use --force to upload anyway.")
        return 1
    
    if not artifacts_list:
        print("\nNo artifacts found to upload.")
        return 1
    
    # Step 2: Create Storage client
    client = create_storage_client()
    if not client:
        return 1
    
    # Step 3: Create bucket if needed
    bucket = create_bucket_if_not_exists(client, args.bucket_name, args.project_id, args.location)
    if not bucket:
        return 1
    
    # Step 4: Upload artifacts
    success, uploaded_files = upload_artifacts(
        client, 
        args.bucket_name, 
        args.artifacts_dir, 
        artifacts_list, 
        args.model_version
    )
    
    if success:
        print(f"\nSUCCESS: Uploaded {len(uploaded_files)} artifacts to GCS")
        print("\nUploaded files:")
        for file_path in uploaded_files:
            print(f"{file_path}")
        
        print(f"\nAccess your models at:")
        print(f"   https://console.cloud.google.com/storage/browser/{args.bucket_name}/models")
        
        return 0
    else:
        print(f"\nFAILED: Upload incomplete")
        return 1


if __name__ == "__main__":
    sys.exit(main())
