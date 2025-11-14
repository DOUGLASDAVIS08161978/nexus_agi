# Google Kubernetes Engine (GKE) Deployment

This directory contains Kubernetes manifests for deploying Nexus AGI to Google Kubernetes Engine.

## Files

- `namespace.yaml` - Creates the nexus-agi namespace
- `deployments.yaml` - Deployment configurations for Nexus AGI and ARIA
- `services.yaml` - LoadBalancer services and persistent volume claims
- `hpa.yaml` - Horizontal Pod Autoscaler configurations

## Quick Start

### Prerequisites

1. **GKE Cluster**: Create a cluster or use existing
```bash
gcloud container clusters create nexus-agi-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10
```

2. **Get Credentials**:
```bash
gcloud container clusters get-credentials nexus-agi-cluster --zone us-central1-a
```

3. **Update Project ID**: Replace `PROJECT_ID` in `deployments.yaml` with your actual GCP project ID

### Create Secrets

```bash
# Create secrets from environment file
kubectl create secret generic nexus-secrets \
  --from-env-file=../.env.gcp \
  -n nexus-agi

# Create GCP service account key secret
kubectl create secret generic gcp-credentials \
  --from-file=key.json=path/to/your-service-account-key.json \
  -n nexus-agi
```

### Deploy

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f deployments.yaml
kubectl apply -f services.yaml
kubectl apply -f hpa.yaml

# Verify deployment
kubectl get pods -n nexus-agi
kubectl get services -n nexus-agi
```

## Configuration

### Resource Limits

Edit `deployments.yaml` to adjust:
- Memory: `requests.memory` and `limits.memory`
- CPU: `requests.cpu` and `limits.cpu`

Default configuration:
- **Nexus AGI**: 4Gi-8Gi RAM, 2-4 CPUs
- **ARIA**: 2Gi-4Gi RAM, 1-2 CPUs

### Autoscaling

Edit `hpa.yaml` to adjust:
- `minReplicas`: Minimum number of pods
- `maxReplicas`: Maximum number of pods
- CPU/Memory thresholds for scaling

### Storage

Persistent storage for Nexus AGI is configured with:
- Access mode: ReadWriteOnce
- Storage class: standard-rwo
- Size: 100Gi (adjustable in `services.yaml`)

## Monitoring

### View Logs

```bash
# Nexus AGI logs
kubectl logs -f deployment/nexus-agi -n nexus-agi

# ARIA logs
kubectl logs -f deployment/aria -n nexus-agi

# All logs
kubectl logs -f -l app=nexus-agi -n nexus-agi
```

### Check Status

```bash
# Pod status
kubectl get pods -n nexus-agi -w

# Service endpoints
kubectl get services -n nexus-agi

# HPA status
kubectl get hpa -n nexus-agi

# Resource usage
kubectl top pods -n nexus-agi
```

### Events

```bash
kubectl get events -n nexus-agi --sort-by='.lastTimestamp'
```

## Scaling

### Manual Scaling

```bash
# Scale Nexus AGI
kubectl scale deployment nexus-agi --replicas=5 -n nexus-agi

# Scale ARIA
kubectl scale deployment aria --replicas=3 -n nexus-agi
```

### Auto-scaling

Horizontal Pod Autoscaler is configured to scale based on:
- CPU utilization: Target 70%
- Memory utilization: Target 80%
- Range: 2-10 pods

## Updating

### Update Image

```bash
# Update to new image
kubectl set image deployment/nexus-agi \
  nexus-agi=gcr.io/PROJECT_ID/nexus-agi:new-tag \
  -n nexus-agi

# Rollout status
kubectl rollout status deployment/nexus-agi -n nexus-agi

# Rollback if needed
kubectl rollout undo deployment/nexus-agi -n nexus-agi
```

### Update Configuration

```bash
# Edit deployment
kubectl edit deployment nexus-agi -n nexus-agi

# Or apply updated YAML
kubectl apply -f deployments.yaml
```

## Troubleshooting

### Pod Not Starting

```bash
# Describe pod
kubectl describe pod <pod-name> -n nexus-agi

# Check events
kubectl get events -n nexus-agi

# Check logs
kubectl logs <pod-name> -n nexus-agi
```

### Common Issues

1. **ImagePullBackOff**: Check image name and GCR permissions
2. **CrashLoopBackOff**: Check logs and environment variables
3. **Pending**: Check resource availability and PVC status

### Debug Commands

```bash
# Execute command in pod
kubectl exec -it <pod-name> -n nexus-agi -- /bin/bash

# Port forward for local testing
kubectl port-forward deployment/nexus-agi 8080:8080 -n nexus-agi

# Copy files from pod
kubectl cp nexus-agi/<pod-name>:/app/logs ./logs -n nexus-agi
```

## Cleanup

```bash
# Delete all resources
kubectl delete namespace nexus-agi

# Or delete individually
kubectl delete -f hpa.yaml
kubectl delete -f services.yaml
kubectl delete -f deployments.yaml
kubectl delete -f namespace.yaml
```

## Advanced

### Custom Health Checks

Health checks are configured in `deployments.yaml`:
- **Liveness Probe**: Restarts pod if fails
- **Readiness Probe**: Removes from service if fails

### Network Policies

To add network policies:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nexus-network-policy
  namespace: nexus-agi
spec:
  podSelector:
    matchLabels:
      app: nexus-agi
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector: {}
  egress:
  - to:
    - podSelector: {}
```

### Resource Quotas

To set namespace quotas:
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: nexus-quota
  namespace: nexus-agi
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
```

## Support

For GKE-specific issues:
- Check [GKE documentation](https://cloud.google.com/kubernetes-engine/docs)
- View [Troubleshooting guide](https://cloud.google.com/kubernetes-engine/docs/troubleshooting)
- See main [DEPLOYMENT_GCP.md](../DEPLOYMENT_GCP.md) for comprehensive guide
