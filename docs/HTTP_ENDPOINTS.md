# HTTP Endpoints

The NAICS MCP Server provides HTTP endpoints for health monitoring, metrics, and status reporting. These endpoints run on a separate port (default: 9090) alongside the MCP server.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NAICS_HTTP_ENABLED` | `true` | Enable/disable HTTP server |
| `NAICS_HTTP_HOST` | `0.0.0.0` | Bind address |
| `NAICS_HTTP_PORT` | `9090` | Port number |
| `NAICS_HEALTH_PATH` | `/health` | Liveness probe path |
| `NAICS_READY_PATH` | `/ready` | Readiness probe path |
| `NAICS_STATUS_PATH` | `/status` | Status endpoint path |
| `NAICS_METRICS_PATH` | `/metrics` | Prometheus metrics path |

## Endpoints

### GET /health

**Purpose**: Liveness probe - is the process alive?

Use this endpoint for Kubernetes `livenessProbe`. Returns 200 if the server process is running.

**Response**:

```json
{
  "status": "alive",
  "timestamp": "2024-01-15T10:30:45.123456+00:00"
}
```

**During Shutdown**:

```json
{
  "status": "alive",
  "timestamp": "2024-01-15T10:30:45.123456+00:00",
  "shutdown_state": "draining"
}
```

**Status Codes**:
- `200` - Server is alive

**Kubernetes Example**:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 9090
  initialDelaySeconds: 30
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3
```

---

### GET /ready

**Purpose**: Readiness probe - is the server ready for traffic?

Use this endpoint for Kubernetes `readinessProbe`. Returns 200 only when all components are initialized and ready to handle requests.

**Response (Ready)**:

```json
{
  "status": "ready",
  "uptime_seconds": 120.5,
  "timestamp": "2024-01-15T10:30:45.123456+00:00"
}
```

**Response (Not Ready - Components Initializing)**:

```json
{
  "status": "not_ready",
  "reason": "components_not_ready",
  "timestamp": "2024-01-15T10:30:45.123456+00:00"
}
```

**Response (Not Ready - Shutting Down)**:

```json
{
  "status": "not_ready",
  "reason": "shutting_down",
  "shutdown_state": "draining",
  "timestamp": "2024-01-15T10:30:45.123456+00:00"
}
```

**Status Codes**:
- `200` - Server is ready for traffic
- `503` - Server is not ready (initializing or shutting down)

**Kubernetes Example**:

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 9090
  initialDelaySeconds: 60
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
  successThreshold: 1
```

---

### GET /status

**Purpose**: Detailed status - comprehensive server status with component health.

Use this endpoint for monitoring dashboards and debugging. Always returns 200 with current status regardless of health.

**Response**:

```json
{
  "version": "0.1.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:30:45.123456+00:00",
  "shutdown": {
    "state": "running",
    "in_flight_requests": 2
  },
  "health": {
    "status": "healthy",
    "components": {
      "database": {
        "status": "ready",
        "total_codes": 2125
      },
      "embedder": {
        "status": "ready",
        "model": "all-MiniLM-L6-v2"
      },
      "search_engine": {
        "status": "ready"
      },
      "embeddings": {
        "status": "ready",
        "coverage_percent": 100.0
      },
      "cross_references": {
        "status": "ready",
        "total_references": 4500
      }
    }
  }
}
```

**Health Status Values**:
- `healthy` - All components ready
- `degraded` - Some components partial, but operational
- `unhealthy` - Critical components failed

**Shutdown State Values**:
- `running` - Normal operation
- `draining` - Waiting for in-flight requests to complete
- `stopping` - Executing shutdown hooks
- `stopped` - Shutdown complete

**Status Codes**:
- `200` - Status retrieved (always)

---

### GET /metrics

**Purpose**: Prometheus metrics endpoint.

Returns metrics in Prometheus text exposition format for scraping by Prometheus.

**Response** (text/plain):

```
# HELP naics_requests_total Total number of MCP requests
# TYPE naics_requests_total counter
naics_requests_total{tool="search_naics_codes",status="success"} 1523.0
naics_requests_total{tool="classify_business",status="success"} 892.0
naics_requests_total{tool="search_naics_codes",status="error"} 3.0

# HELP naics_request_duration_seconds Request duration in seconds
# TYPE naics_request_duration_seconds histogram
naics_request_duration_seconds_bucket{tool="search_naics_codes",le="0.05"} 1200.0
naics_request_duration_seconds_bucket{tool="search_naics_codes",le="0.1"} 1450.0
naics_request_duration_seconds_bucket{tool="search_naics_codes",le="0.25"} 1510.0
naics_request_duration_seconds_bucket{tool="search_naics_codes",le="0.5"} 1520.0
naics_request_duration_seconds_bucket{tool="search_naics_codes",le="1.0"} 1523.0
naics_request_duration_seconds_bucket{tool="search_naics_codes",le="+Inf"} 1523.0
naics_request_duration_seconds_count{tool="search_naics_codes"} 1523.0
naics_request_duration_seconds_sum{tool="search_naics_codes"} 98.45

# HELP naics_active_requests Current number of in-flight requests
# TYPE naics_active_requests gauge
naics_active_requests 2.0

# HELP naics_search_results_count Number of results returned per search
# TYPE naics_search_results_count histogram
naics_search_results_count_bucket{strategy="hybrid",le="1.0"} 50.0
naics_search_results_count_bucket{strategy="hybrid",le="5.0"} 800.0
naics_search_results_count_bucket{strategy="hybrid",le="10.0"} 1500.0
naics_search_results_count_bucket{strategy="hybrid",le="+Inf"} 1523.0

# HELP naics_embeddings_total Total embeddings loaded
# TYPE naics_embeddings_total gauge
naics_embeddings_total 2125.0

# HELP naics_database_codes_total Total NAICS codes in database
# TYPE naics_database_codes_total gauge
naics_database_codes_total 2125.0
```

**Status Codes**:
- `200` - Metrics retrieved
- `500` - Error generating metrics

**Prometheus Scrape Config**:

```yaml
scrape_configs:
  - job_name: 'naics-mcp-server'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

---

## Docker Usage

The Docker image exposes port 9090 and includes a health check:

```dockerfile
EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1
```

**Docker Compose**:

```yaml
services:
  naics-mcp-server:
    image: naics-mcp-server:latest
    ports:
      - "9090:9090"  # HTTP endpoints
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
      interval: 30s
      timeout: 5s
      start_period: 60s
      retries: 3
```

---

## Kubernetes Usage

See [k8s/README.md](../k8s/README.md) for full Kubernetes deployment instructions.

**Key Probe Configuration**:

```yaml
# Liveness - is the process alive?
livenessProbe:
  httpGet:
    path: /health
    port: 9090
  initialDelaySeconds: 30
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3

# Readiness - is it ready for traffic?
readinessProbe:
  httpGet:
    path: /ready
    port: 9090
  initialDelaySeconds: 60
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

# Startup - allow time for initial model download
startupProbe:
  httpGet:
    path: /health
    port: 9090
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 30  # 5 minutes total
```

---

## Monitoring Integration

### Prometheus + Grafana

1. Add scrape target to Prometheus:

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

2. Import Grafana dashboard (or create panels for key metrics):
   - Request rate: `rate(naics_requests_total[5m])`
   - Error rate: `rate(naics_requests_total{status="error"}[5m])`
   - Latency p95: `histogram_quantile(0.95, rate(naics_request_duration_seconds_bucket[5m]))`
   - Active requests: `naics_active_requests`

### Alerting Examples

```yaml
groups:
  - name: naics-mcp-server
    rules:
      - alert: HighErrorRate
        expr: rate(naics_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on NAICS MCP Server"

      - alert: SlowRequests
        expr: histogram_quantile(0.95, rate(naics_request_duration_seconds_bucket[5m])) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "NAICS MCP Server requests are slow"

      - alert: ServerNotReady
        expr: up{job="naics-mcp-server"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "NAICS MCP Server is down"
```
