# Google Cloud Platform Setup Guide

This guide walks you through setting up the Alzearly pipeline on Google Cloud Platform (GCP), including BigQuery data upload and querying.

## Prerequisites

1. **Create a Google Cloud Project** in [Google Cloud Console](https://console.cloud.google.com/)
2. **Add Dependencies** to your requirements.txt:
   ```
   google-cloud-bigquery
   google-cloud-core
   ```

## Step 1: Install Google Cloud CLI

### Option A: Using Package Manager (Recommended)
```bash
# Add Google Cloud SDK repository
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import Google's public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Update and install
apt-get update && apt-get install google-cloud-cli
```

### Option B: Direct Installation
```bash
# Install curl if needed
apt-get update && apt-get install -y curl

# Download and install
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

## Step 2: Authentication & Configuration

### Basic Setup
```bash
# Verify installation
gcloud version

# Initialize gcloud (first time)
gcloud init

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Set your account
gcloud config set account YOUR_EMAIL@gmail.com
```

### Authentication Method 1: User Authentication
```bash
# Authenticate for BigQuery access
gcloud auth application-default login

# Verify authentication
gcloud auth list
gcloud config list
```

### Authentication Method 2: Service Account (Recommended for Production)

#### Create Service Account:
1. Go to [Google Cloud Console](https://console.cloud.google.com/) → **IAM & Admin** → **Service Accounts**
2. Click **"+ CREATE SERVICE ACCOUNT"**
3. Fill in details:
   - **Name**: `bigquery-uploader`
   - **Description**: `Service account for BigQuery data uploads`
4. Click **"CREATE AND CONTINUE"**

#### Grant Permissions:
Add these roles:
- **BigQuery Admin** (for full BigQuery access)
- **Storage Object Viewer** (if reading from Cloud Storage)
- **Storage Object Creator** (if writing to Cloud Storage)

#### Download Key:
1. Find your service account in the list
2. Click on the service account name
3. Go to **"KEYS"** tab
4. Click **"ADD KEY"** → **"Create new key"**
5. Select **"JSON"** format
6. Click **"CREATE"**
7. Move the downloaded key to your workspace

#### Use Service Account:
```bash
# Authenticate with service account
gcloud auth activate-service-account --key-file=/workspace/your-service-account-key.json

# Or set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/your-service-account-key.json"
```

## Step 3: Test & Verify Setup

```bash
# Test authentication
gcloud auth list

# Test project access
gcloud projects describe YOUR_PROJECT_ID

# Test BigQuery access
bq ls --project_id=YOUR_PROJECT_ID
```

## Step 4: Upload Data to BigQuery

### Prerequisites: 
Ensure you have generated data first:
```bash
# Generate data if needed
make gen-data-tiny  # or make gen-data
```

### Upload to BigQuery:
```bash
python src/cli/upload_bigQuery.py --data-path /Data/featurized --project-id YOUR_PROJECT_ID
```

### Expected Output:
```
Created dataset: YOUR_PROJECT_ID.alzearly
Uploading to BigQuery...
Table: YOUR_PROJECT_ID.alzearly.patient_features

Uploaded: data.parquet
SUCCESS: Loaded X,XXX rows into YOUR_PROJECT_ID.alzearly.patient_features
```

## Step 5: View Data in BigQuery

### Using Web Console:
1. Go to [BigQuery Console](https://console.cloud.google.com/bigquery)
2. Navigate to your project → `alzearly` dataset → `patient_features` table
3. Click **"Preview"** to see sample data
4. Use **"Query"** to run SQL queries

### Sample Queries:
```sql
-- View sample data
SELECT * FROM `YOUR_PROJECT_ID.alzearly.patient_features` LIMIT 10;

-- Count total rows
SELECT COUNT(*) as total_rows 
FROM `YOUR_PROJECT_ID.alzearly.patient_features`;

-- Basic statistics
SELECT 
  COUNT(*) as total_patients,
  COUNT(DISTINCT patient_id) as unique_patients,
  MIN(year) as min_year,
  MAX(year) as max_year,
  AVG(age) as avg_age,
  SUM(CASE WHEN alzheimers_diagnosis = true THEN 1 ELSE 0 END) as positive_cases
FROM `YOUR_PROJECT_ID.alzearly.patient_features`;

-- View table schema
SELECT column_name, data_type 
FROM `YOUR_PROJECT_ID.alzearly.INFORMATION_SCHEMA.COLUMNS` 
WHERE table_name = 'patient_features';
```

### Using Command Line:
```bash
# List datasets
bq ls --project_id=YOUR_PROJECT_ID

# Show table schema
bq show YOUR_PROJECT_ID:alzearly.patient_features

# Query data
bq query --use_legacy_sql=false \
"SELECT * FROM \`YOUR_PROJECT_ID.alzearly.patient_features\` LIMIT 10"
```

## Troubleshooting

### Authentication Issues:
```bash
# Clear all authentication and start fresh
gcloud auth revoke --all
gcloud auth login
gcloud auth application-default login

# Run diagnostics
gcloud info --run-diagnostics

# Check active accounts
gcloud auth list --filter=status:ACTIVE --format="table(account)"
```

### Common Errors:

| Error | Solution |
|-------|----------|
| `gcloud: command not found` | Install Google Cloud CLI |
| `Authentication error` | Run `gcloud auth application-default login` |
| `Project not found` | Check project ID and permissions |
| `Data path does not exist` | Generate data first with `make gen-data-tiny` |
| `Permission denied` | Ensure BigQuery Admin role is assigned |
| `Scope has changed` | Clear auth and re-authenticate |

### Debugging Commands:
```bash
# Check current configuration
gcloud config list

# Check authentication status
gcloud auth list

# Test BigQuery permissions
bq ls --project_id=YOUR_PROJECT_ID

# Verify service account key
gcloud auth activate-service-account --key-file=YOUR_KEY_FILE.json --dry-run
```

## Security Best Practices

1. **Service Account Keys:**
   - Keep JSON key files secure and never commit to version control
   - Use minimal required permissions
   - Rotate keys regularly
   - Consider using Workload Identity for production

2. **Access Control:**
   - Use principle of least privilege
   - Regularly audit IAM permissions
   - Enable audit logging

3. **Data Protection:**
   - Enable encryption at rest and in transit
   - Use VPC networks for enhanced security
   - Implement proper backup strategies

## Next Steps

After successful setup:
1. **Explore your data** using BigQuery's web interface
2. **Create dashboards** using Google Data Studio
3. **Set up Cloud Scheduler** for automated data pipeline runs
4. **Configure Cloud Functions** for event-driven processing
5. **Implement CI/CD** using Cloud Build

For more advanced configurations, see the [Google Cloud Documentation](https://cloud.google.com/docs).
