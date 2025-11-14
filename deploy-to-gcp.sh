#!/bin/bash
# Quick deployment script for Google Cloud Platform
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    print_success "gcloud CLI found"
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker found"
}

# Get configuration
get_config() {
    print_info "Configuring deployment..."
    
    # Get current project
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    
    if [ -z "$PROJECT_ID" ]; then
        read -p "Enter your GCP Project ID: " PROJECT_ID
        gcloud config set project $PROJECT_ID
    fi
    
    print_success "Using project: $PROJECT_ID"
    
    # Get region
    read -p "Enter region (default: us-central1): " REGION
    REGION=${REGION:-us-central1}
    print_success "Using region: $REGION"
    
    # Deployment type
    echo ""
    echo "Select deployment type:"
    echo "1) Cloud Run (Serverless, recommended)"
    echo "2) Google Kubernetes Engine (GKE)"
    echo "3) Both Cloud Run and GKE"
    read -p "Enter choice [1-3]: " DEPLOY_TYPE
}

# Enable APIs
enable_apis() {
    print_info "Enabling required GCP APIs..."
    
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        container.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com \
        sqladmin.googleapis.com \
        secretmanager.googleapis.com \
        --quiet
    
    print_success "APIs enabled"
}

# Build and push images
build_images() {
    print_info "Building Docker images..."
    
    # Configure Docker
    gcloud auth configure-docker --quiet
    
    # Build Nexus image
    print_info "Building Nexus AGI image..."
    docker build -f Dockerfile.nexus.gcp -t gcr.io/${PROJECT_ID}/nexus-agi:latest .
    print_success "Nexus AGI image built"
    
    # Build ARIA image
    print_info "Building ARIA image..."
    docker build -f Dockerfile.aria.gcp -t gcr.io/${PROJECT_ID}/aria:latest .
    print_success "ARIA image built"
    
    # Push images
    print_info "Pushing images to Google Container Registry..."
    docker push gcr.io/${PROJECT_ID}/nexus-agi:latest
    docker push gcr.io/${PROJECT_ID}/aria:latest
    print_success "Images pushed to GCR"
}

# Deploy to Cloud Run
deploy_cloud_run() {
    print_info "Deploying to Cloud Run..."
    
    # Deploy Nexus AGI
    print_info "Deploying Nexus AGI service..."
    gcloud run deploy nexus-agi \
        --image gcr.io/${PROJECT_ID}/nexus-agi:latest \
        --platform managed \
        --region ${REGION} \
        --memory 4Gi \
        --cpu 2 \
        --timeout 3600 \
        --max-instances 10 \
        --allow-unauthenticated \
        --quiet
    
    NEXUS_URL=$(gcloud run services describe nexus-agi --region ${REGION} --format 'value(status.url)')
    print_success "Nexus AGI deployed: $NEXUS_URL"
    
    # Deploy ARIA
    print_info "Deploying ARIA service..."
    gcloud run deploy aria \
        --image gcr.io/${PROJECT_ID}/aria:latest \
        --platform managed \
        --region ${REGION} \
        --memory 2Gi \
        --cpu 1 \
        --timeout 3600 \
        --max-instances 10 \
        --allow-unauthenticated \
        --quiet
    
    ARIA_URL=$(gcloud run services describe aria --region ${REGION} --format 'value(status.url)')
    print_success "ARIA deployed: $ARIA_URL"
}

# Deploy to GKE
deploy_gke() {
    print_info "Deploying to Google Kubernetes Engine..."
    
    # Check if cluster exists
    if ! gcloud container clusters describe nexus-agi-cluster --zone ${REGION}-a &>/dev/null; then
        print_info "Creating GKE cluster (this may take 5-10 minutes)..."
        gcloud container clusters create nexus-agi-cluster \
            --zone ${REGION}-a \
            --num-nodes 3 \
            --machine-type n1-standard-4 \
            --enable-autoscaling \
            --min-nodes 1 \
            --max-nodes 10 \
            --enable-autorepair \
            --enable-autoupgrade \
            --quiet
        print_success "GKE cluster created"
    else
        print_success "Using existing GKE cluster"
    fi
    
    # Get credentials
    gcloud container clusters get-credentials nexus-agi-cluster --zone ${REGION}-a --quiet
    print_success "Cluster credentials configured"
    
    # Update deployments with project ID
    sed -i.bak "s/PROJECT_ID/${PROJECT_ID}/g" gke/deployments.yaml
    
    # Deploy to GKE
    print_info "Deploying applications to GKE..."
    kubectl apply -f gke/namespace.yaml
    kubectl apply -f gke/deployments.yaml
    kubectl apply -f gke/services.yaml
    kubectl apply -f gke/hpa.yaml
    
    # Restore original file
    mv gke/deployments.yaml.bak gke/deployments.yaml
    
    print_success "Applications deployed to GKE"
    
    # Get service endpoints
    print_info "Waiting for LoadBalancer IPs..."
    sleep 30
    
    NEXUS_IP=$(kubectl get service nexus-agi-service -n nexus-agi -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    ARIA_IP=$(kubectl get service aria-service -n nexus-agi -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$NEXUS_IP" != "pending" ]; then
        print_success "Nexus AGI endpoint: http://${NEXUS_IP}"
    else
        print_info "Nexus AGI endpoint: pending (check with: kubectl get svc -n nexus-agi)"
    fi
    
    if [ "$ARIA_IP" != "pending" ]; then
        print_success "ARIA endpoint: http://${ARIA_IP}"
    else
        print_info "ARIA endpoint: pending (check with: kubectl get svc -n nexus-agi)"
    fi
}

# Main deployment flow
main() {
    echo ""
    echo "=========================================="
    echo "  Nexus AGI - Google Cloud Deployment"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    get_config
    enable_apis
    build_images
    
    case $DEPLOY_TYPE in
        1)
            deploy_cloud_run
            ;;
        2)
            deploy_gke
            ;;
        3)
            deploy_cloud_run
            deploy_gke
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "  Deployment Complete!"
    echo "=========================================="
    echo ""
    
    if [ "$DEPLOY_TYPE" == "1" ] || [ "$DEPLOY_TYPE" == "3" ]; then
        echo "Cloud Run Services:"
        echo "  Nexus AGI: $NEXUS_URL"
        echo "  ARIA: $ARIA_URL"
        echo ""
    fi
    
    if [ "$DEPLOY_TYPE" == "2" ] || [ "$DEPLOY_TYPE" == "3" ]; then
        echo "GKE Services:"
        echo "  Check status: kubectl get pods -n nexus-agi"
        echo "  View logs: kubectl logs -f deployment/nexus-agi -n nexus-agi"
        echo ""
    fi
    
    echo "Next steps:"
    echo "  1. Configure environment variables in .env.gcp"
    echo "  2. Set up Secret Manager for sensitive data"
    echo "  3. Configure Cloud SQL or external databases"
    echo "  4. Review and adjust resource limits"
    echo "  5. Set up monitoring and alerts"
    echo ""
    echo "For detailed documentation, see DEPLOYMENT_GCP.md"
    echo ""
}

# Run main function
main
