# Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the NAICS MCP Server.

## Quick Start

```bash
# Deploy base configuration
kubectl apply -k k8s/base/

# Or deploy production overlay
kubectl apply -k k8s/overlays/production/
```

## Directory Structure

```
k8s/
├── base/                    # Base configuration
│   ├── namespace.yaml       # naics-mcp namespace
│   ├── configmap.yaml       # Application configuration
│   ├── pvc.yaml             # Persistent storage (data + cache)
│   ├── deployment.yaml      # Deployment with health probes
│   ├── service.yaml         # ClusterIP + headless services
│   ├── servicemonitor.yaml  # Prometheus Operator monitoring
│   └── kustomization.yaml   # Kustomize configuration
└── overlays/
    └── production/          # Production overrides
        └── kustomization.yaml
```

## Components

### Namespace
Creates `naics-mcp` namespace for isolation.

### ConfigMap
Non-sensitive configuration:
- Database path
- Logging settings
- HTTP server configuration
- Search parameters

### PersistentVolumeClaims
- `naics-mcp-data` (1Gi): DuckDB database storage
- `naics-mcp-cache` (2Gi): Sentence-transformers model cache

### Deployment
- Single replica (scale in production overlay)
- Non-root security context
- Three probe types:
  - **Startup**: Allows up to 5 minutes for initial model download
  - **Liveness**: Confirms process is alive (`/health`)
  - **Readiness**: Confirms ready for traffic (`/ready`)
- Resource limits (1Gi-4Gi memory, 0.5-2 CPU)

### Service
- `naics-mcp-server`: ClusterIP on port 9090
- `naics-mcp-server-headless`: For direct pod access

### ServiceMonitor
Prometheus Operator integration for automatic metrics scraping.

## Customization

### Using Overlays

Create environment-specific overlays:

```bash
mkdir -p k8s/overlays/staging
```

```yaml
# k8s/overlays/staging/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../base

commonLabels:
  environment: staging

images:
  - name: naics-mcp-server
    newTag: staging-latest
```

### Image Configuration

Update the image in your overlay:

```yaml
images:
  - name: naics-mcp-server
    newName: your-registry.io/naics-mcp-server
    newTag: v1.0.0
```

### Resource Tuning

Patch resources based on your needs:

```yaml
patches:
  - patch: |-
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: naics-mcp-server
      spec:
        template:
          spec:
            containers:
              - name: naics-mcp-server
                resources:
                  limits:
                    memory: "8Gi"
```

## Monitoring

### Prometheus Operator

If using Prometheus Operator, uncomment the ServiceMonitor in `base/kustomization.yaml`:

```yaml
resources:
  # ...
  - servicemonitor.yaml
```

### Manual Prometheus

Add scrape config:

```yaml
scrape_configs:
  - job_name: 'naics-mcp-server'
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names: ['naics-mcp']
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_label_app_kubernetes_io_name]
        regex: naics-mcp-server
        action: keep
```

## Troubleshooting

### Check Pod Status
```bash
kubectl -n naics-mcp get pods
kubectl -n naics-mcp describe pod <pod-name>
```

### View Logs
```bash
kubectl -n naics-mcp logs -f deployment/naics-mcp-server
```

### Check Health
```bash
kubectl -n naics-mcp port-forward svc/naics-mcp-server 9090:9090
curl http://localhost:9090/health
curl http://localhost:9090/ready
curl http://localhost:9090/metrics
```

### Debug Storage
```bash
kubectl -n naics-mcp get pvc
kubectl -n naics-mcp exec -it deployment/naics-mcp-server -- ls -la /app/data
```
