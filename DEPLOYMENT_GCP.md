# Google Cloud Platform Deployment Guide

This guide explains how to deploy the Nexus AGI and ARIA systems to Google Cloud Platform (GCP).

## Deployment Options

1. **Cloud Run** - Serverless, fully managed (Recommended for getting started)
2. **Google Kubernetes Engine (GKE)** - For advanced orchestration
3. **Compute Engine** - Traditional VM deployment
4. **Cloud Functions** - Event-driven execution

## Prerequisites

### Required Tools
- [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/install)
- [Docker](https://docs.docker.com/get-docker/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) (for GKE)
- [Terraform](https://www.terraform.io/downloads) (optional, for IaC)

### GCP Setup
```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  container.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com
```

## Option 1: Cloud Run Deployment (Recommended)

Cloud Run is the easiest way to deploy containerized applications on GCP.

### Step 1: Prepare Environment Variables

Create a `.env.gcp` file:
```bash
# API Keys
GITHUB_TOKEN=your_github_token
GITLAB_TOKEN=your_gitlab_token
HUGGINGFACE_TOKEN=your_hf_token
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database (Cloud SQL or external)
POSTGRES_HOST=your_cloud_sql_ip
POSTGRES_DB=nexus
POSTGRES_USER=nexus
POSTGRES_PASSWORD=your_password
MONGODB_URI=mongodb://your_mongodb_uri
REDIS_HOST=your_redis_ip

# Google Cloud Storage
GCS_BUCKET=nexus-agi-storage
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-credentials.json

# Communication
SENDGRID_API_KEY=your_sendgrid_key
SLACK_BOT_TOKEN=your_slack_token
DISCORD_BOT_TOKEN=your_discord_token
```

### Step 2: Build and Push Docker Images

```bash
# Set variables
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1

# Configure Docker for GCP
gcloud auth configure-docker

# Build Nexus image
docker build -f Dockerfile.nexus.gcp -t gcr.io/${PROJECT_ID}/nexus-agi:latest .

# Build ARIA image
docker build -f Dockerfile.aria.gcp -t gcr.io/${PROJECT_ID}/aria:latest .

# Push to Google Container Registry
docker push gcr.io/${PROJECT_ID}/nexus-agi:latest
docker push gcr.io/${PROJECT_ID}/aria:latest
```

### Step 3: Deploy to Cloud Run

```bash
# Deploy Nexus AGI
gcloud run deploy nexus-agi \
  --image gcr.io/${PROJECT_ID}/nexus-agi:latest \
  --platform managed \
  --region ${REGION} \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --env-vars-file .env.gcp \
  --allow-unauthenticated

# Deploy ARIA
gcloud run deploy aria \
  --image gcr.io/${PROJECT_ID}/aria:latest \
  --platform managed \
  --region ${REGION} \
  --memory 2Gi \
  --cpu 1 \
  --timeout 3600 \
  --max-instances 10 \
  --env-vars-file .env.gcp \
  --allow-unauthenticated
```

### Step 4: Configure Cloud Scheduler (for continuous execution)

```bash
# Create scheduler jobs
gcloud scheduler jobs create http nexus-trigger \
  --location ${REGION} \
  --schedule "*/5 * * * *" \
  --uri "https://nexus-agi-[hash]-uc.a.run.app/execute" \
  --http-method POST \
  --oidc-service-account-email nexus-sa@${PROJECT_ID}.iam.gserviceaccount.com

gcloud scheduler jobs create http aria-trigger \
  --location ${REGION} \
  --schedule "*/5 * * * *" \
  --uri "https://aria-[hash]-uc.a.run.app/execute" \
  --http-method POST \
  --oidc-service-account-email aria-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

## Option 2: Google Kubernetes Engine (GKE)

For production workloads requiring advanced orchestration.

### Step 1: Create GKE Cluster

```bash
# Create cluster
gcloud container clusters create nexus-agi-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials nexus-agi-cluster --zone us-central1-a

# Verify connection
kubectl cluster-info
```

### Step 2: Create Kubernetes Secrets

```bash
# Create secret from .env.gcp
kubectl create secret generic nexus-secrets --from-env-file=.env.gcp

# Create GCP service account key secret
kubectl create secret generic gcp-credentials \
  --from-file=key.json=path/to/your-service-account-key.json
```

### Step 3: Deploy to GKE

```bash
# Apply Kubernetes configurations
kubectl apply -f gke/namespace.yaml
kubectl apply -f gke/nexus-deployment.yaml
kubectl apply -f gke/aria-deployment.yaml
kubectl apply -f gke/services.yaml

# Check deployment status
kubectl get pods -n nexus-agi
kubectl get services -n nexus-agi

# View logs
kubectl logs -f deployment/nexus-agi -n nexus-agi
kubectl logs -f deployment/aria -n nexus-agi
```

### Step 4: Setup Autoscaling

```bash
# Horizontal Pod Autoscaling
kubectl apply -f gke/hpa.yaml

# Verify HPA
kubectl get hpa -n nexus-agi
```

## Option 3: Compute Engine VM

Traditional VM deployment for full control.

### Step 1: Create VM Instance

```bash
gcloud compute instances create nexus-agi-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --tags=nexus-agi \
  --metadata-from-file startup-script=gce/startup-script.sh
```

### Step 2: SSH and Setup

```bash
# SSH into instance
gcloud compute ssh nexus-agi-vm --zone us-central1-a

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Clone repository
git clone https://github.com/DOUGLASDAVIS08161978/nexus_agi.git
cd nexus_agi

# Copy environment variables
vi .env  # Add your environment variables

# Deploy with Docker Compose
docker-compose up -d
```

## Infrastructure as Code (Terraform)

### Step 1: Initialize Terraform

```bash
cd terraform/gcp

# Initialize
terraform init

# Plan deployment
terraform plan -var="project_id=YOUR_PROJECT_ID"

# Apply configuration
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

### Step 2: Terraform Configuration

See `terraform/gcp/main.tf` for complete infrastructure definition including:
- VPC and networking
- GKE cluster
- Cloud SQL databases
- Cloud Storage buckets
- IAM roles and service accounts
- Secret Manager
- Cloud Monitoring

## Cloud Storage Setup

### Create Storage Buckets

```bash
# Create main storage bucket
gsutil mb -p ${PROJECT_ID} -c STANDARD -l ${REGION} gs://nexus-agi-storage

# Create backup bucket
gsutil mb -p ${PROJECT_ID} -c NEARLINE -l ${REGION} gs://nexus-agi-backups

# Set lifecycle policy
gsutil lifecycle set gcs/lifecycle-policy.json gs://nexus-agi-storage
```

### Configure Access

```bash
# Grant service account access
gsutil iam ch \
  serviceAccount:nexus-sa@${PROJECT_ID}.iam.gserviceaccount.com:objectAdmin \
  gs://nexus-agi-storage
```

## Database Setup (Cloud SQL)

### Create PostgreSQL Instance

```bash
gcloud sql instances create nexus-postgres \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=${REGION} \
  --root-password=your_secure_password \
  --storage-size=10GB \
  --storage-type=SSD \
  --backup

# Create database
gcloud sql databases create nexus --instance=nexus-postgres

# Create user
gcloud sql users create nexus \
  --instance=nexus-postgres \
  --password=your_secure_password
```

### Create MongoDB (via Google Marketplace or external)

For MongoDB, consider:
- MongoDB Atlas (managed MongoDB on GCP)
- Google Marketplace MongoDB solution
- Self-managed on Compute Engine

## Monitoring & Logging

### Cloud Monitoring Setup

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=nexus-agi" \
  --limit 50 \
  --format json

# Create log-based metrics
gcloud logging metrics create nexus_errors \
  --description="Nexus AGI error count" \
  --log-filter='resource.type="cloud_run_revision" AND severity="ERROR"'

# Create alerts
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="Nexus High Error Rate" \
  --condition-display-name="Error rate > 10/min" \
  --condition-threshold-value=10 \
  --condition-threshold-duration=60s
```

### Cloud Trace

```bash
# Enable Cloud Trace
gcloud services enable cloudtrace.googleapis.com

# View traces
gcloud trace list --project=${PROJECT_ID}
```

## Cost Optimization

### Recommendations

1. **Use Preemptible VMs** for non-critical workloads:
```bash
gcloud compute instances create nexus-preemptible \
  --preemptible \
  --machine-type=n1-standard-2
```

2. **Set budget alerts**:
```bash
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Nexus AGI Budget" \
  --budget-amount=100USD
```

3. **Use autoscaling** to scale down during low usage
4. **Use Cloud Storage lifecycle policies** for old data
5. **Monitor costs** in Cloud Console

## Security Best Practices

### IAM Configuration

```bash
# Create service account
gcloud iam service-accounts create nexus-sa \
  --display-name="Nexus AGI Service Account"

# Grant minimal required permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:nexus-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:nexus-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"
```

### Secret Management

```bash
# Store secrets in Secret Manager
echo -n "your_api_key" | gcloud secrets create github-token \
  --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding github-token \
  --member="serviceAccount:nexus-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Network Security

```bash
# Create firewall rules
gcloud compute firewall-rules create allow-nexus-agi \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags nexus-agi

# Use VPC for internal communication
gcloud compute networks create nexus-vpc --subnet-mode=custom
```

## Backup & Disaster Recovery

### Automated Backups

```bash
# Cloud SQL automatic backups (already enabled)
gcloud sql backups list --instance=nexus-postgres

# Manual backup
gcloud sql backups create --instance=nexus-postgres

# Storage bucket backup
gsutil -m rsync -r gs://nexus-agi-storage gs://nexus-agi-backups
```

### Disaster Recovery Plan

1. **Database**: Point-in-time recovery with Cloud SQL
2. **Storage**: Cross-region replication
3. **Code**: GitHub repository
4. **Infrastructure**: Terraform state backup

## Scaling Guidelines

### Vertical Scaling (Cloud Run)

```bash
# Increase resources
gcloud run services update nexus-agi \
  --memory 8Gi \
  --cpu 4
```

### Horizontal Scaling (GKE)

```bash
# Update replica count
kubectl scale deployment nexus-agi --replicas=5 -n nexus-agi

# Or use HPA (already configured)
kubectl autoscale deployment nexus-agi \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n nexus-agi
```

## Troubleshooting

### Common Issues

**1. Image pull errors:**
```bash
# Verify image exists
gcloud container images list --repository=gcr.io/${PROJECT_ID}

# Check permissions
gcloud projects get-iam-policy ${PROJECT_ID}
```

**2. Out of memory:**
```bash
# Increase memory allocation
gcloud run services update nexus-agi --memory 8Gi
```

**3. Connection timeout:**
```bash
# Increase timeout
gcloud run services update nexus-agi --timeout 3600
```

**4. Database connection issues:**
```bash
# Check Cloud SQL instance
gcloud sql instances describe nexus-postgres

# Test connection
gcloud sql connect nexus-postgres --user=nexus
```

### Debugging

```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" --limit 100

# Describe service
gcloud run services describe nexus-agi --region ${REGION}

# Check events (GKE)
kubectl get events -n nexus-agi --sort-by='.lastTimestamp'
```

## Performance Optimization

### Cloud CDN

```bash
# Enable Cloud CDN for static content
gcloud compute backend-services update nexus-backend \
  --enable-cdn \
  --global
```

### Cloud Load Balancing

```bash
# Create load balancer
gcloud compute forwarding-rules create nexus-lb \
  --global \
  --target-http-proxy=nexus-proxy \
  --ports=80
```

## Cleanup

### Remove All Resources

```bash
# Cloud Run
gcloud run services delete nexus-agi --region ${REGION}
gcloud run services delete aria --region ${REGION}

# GKE
gcloud container clusters delete nexus-agi-cluster --zone us-central1-a

# Cloud SQL
gcloud sql instances delete nexus-postgres

# Storage
gsutil -m rm -r gs://nexus-agi-storage
gsutil -m rm -r gs://nexus-agi-backups

# Or use Terraform
cd terraform/gcp
terraform destroy -var="project_id=YOUR_PROJECT_ID"
```

## Quick Deploy Script

Save as `deploy-to-gcp.sh`:

```bash
#!/bin/bash
set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1

echo "Deploying Nexus AGI to Google Cloud Platform..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Build images
echo "Building Docker images..."
docker build -f Dockerfile.nexus.gcp -t gcr.io/${PROJECT_ID}/nexus-agi:latest .
docker build -f Dockerfile.aria.gcp -t gcr.io/${PROJECT_ID}/aria:latest .

# Push images
echo "Pushing images to GCR..."
docker push gcr.io/${PROJECT_ID}/nexus-agi:latest
docker push gcr.io/${PROJECT_ID}/aria:latest

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy nexus-agi \
  --image gcr.io/${PROJECT_ID}/nexus-agi:latest \
  --platform managed \
  --region ${REGION} \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated

gcloud run deploy aria \
  --image gcr.io/${PROJECT_ID}/aria:latest \
  --platform managed \
  --region ${REGION} \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated

echo "Deployment complete!"
echo "Nexus AGI URL: $(gcloud run services describe nexus-agi --region ${REGION} --format 'value(status.url)')"
echo "ARIA URL: $(gcloud run services describe aria --region ${REGION} --format 'value(status.url)')"
```

Make executable and run:
```bash
chmod +x deploy-to-gcp.sh
./deploy-to-gcp.sh
```

## Support

For issues specific to GCP deployment:
1. Check [GCP documentation](https://cloud.google.com/docs)
2. View [Cloud Run troubleshooting](https://cloud.google.com/run/docs/troubleshooting)
3. Use [GCP support](https://cloud.google.com/support)
4. Open an issue on GitHub with `[GCP]` prefix

## Summary

You can deploy Nexus AGI to GCP using:
- **Cloud Run**: Fastest, serverless, auto-scaling
- **GKE**: Advanced orchestration, full Kubernetes features
- **Compute Engine**: Full VM control, traditional deployment

Choose based on your requirements for control, scalability, and operational complexity.
