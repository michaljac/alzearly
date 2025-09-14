#!/usr/bin/env python3
"""
Upload featurized Parquet data to BigQuery.
Supports both direct table loading and external table creation.
"""

import argparse
import sys
from pathlib import Path
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def create_bigquery_client():
    """Create BigQuery client with authentication."""
    try:
        client = bigquery.Client()
        return client
    except Exception as e:
        print(f"ERROR: Failed to create BigQuery client: {e}")
        print("Make sure you're authenticated with: gcloud auth application-default login")
        return None


def upload_parquet_to_table(client, parquet_path, table_id, write_mode="WRITE_TRUNCATE"):
    """Upload Parquet file(s) directly to BigQuery table."""
    
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=getattr(bigquery.WriteDisposition, write_mode),
        autodetect=True,  # Auto-detect schema from Parquet
    )
    
    try:
        parquet_files = list(parquet_path.glob("*.parquet"))
        if not parquet_files:
            print(f"ERROR: No .parquet files found in {parquet_path}")
            return False
        
        # Upload all files
        for i, file_path in enumerate(parquet_files):
            if i > 0:
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            
            with open(file_path, "rb") as source_file:
                job = client.load_table_from_file(source_file, table_id, job_config=job_config)
                job.result()  # Wait for completion
                print(f"Uploaded: {file_path.name}")
        
        table = client.get_table(table_id)
        print(f"SUCCESS: Loaded {table.num_rows:,} rows into {table_id}")
        return True
        
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload featurized data to BigQuery")
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="../Data/featurized",
        help="Path to featurized Parquet data"
    )
    
    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="GCP Project ID"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="alzearly",
        help="BigQuery dataset name"
    )
    
    parser.add_argument(
        "--table",
        type=str,
        default="patient_features",
        help="BigQuery table name"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Data path {data_path} does not exist")
        return 1
    
    # Create BigQuery client
    client = create_bigquery_client()
    if not client:
        return 1
    
    # Create dataset if it doesn't exist
    dataset_id = f"{args.project_id}.{args.dataset}"
    try:
        client.get_dataset(dataset_id)
    except NotFound:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"  # Change to your preferred location
        client.create_dataset(dataset)
        print(f"Created dataset: {dataset_id}")
    
    # Full table ID
    table_id = f"{args.project_id}.{args.dataset}.{args.table}"
    
    print(f"Uploading to BigQuery...")
    print(f"Table: {table_id}")
    print()
    
    # Upload data
    success = upload_parquet_to_table(client, data_path, table_id)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())