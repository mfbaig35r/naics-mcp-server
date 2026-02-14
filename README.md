# NAICS MCP Server

[![CI](https://github.com/your-org/naics-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/naics-mcp-server/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent industry classification service for NAICS 2022, built with the Model Context Protocol (MCP).

## Features

- **Semantic Search** - Natural language search using sentence embeddings
- **Hybrid Search** - Combines semantic understanding with exact term matching
- **5-Level Hierarchy** - Navigate from Sector (2-digit) to National Industry (6-digit)
- **Cross-Reference Integration** - Critical exclusion checks for accurate classification
- **Index Term Search** - 20,398 official NAICS index terms
- **Classification Workbook** - Record and track classification decisions
- **Health Monitoring** - Kubernetes-ready liveness and readiness probes

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Docker](#docker)
- [MCP Tools](#mcp-tools)
- [CLI Usage](#cli-usage)
- [Health Checks](#health-checks)
- [Development](#development)
- [Architecture](#architecture)
- [License](#license)

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/naics-mcp-server.git
cd naics-mcp-server

# Start the server
docker compose up
```

### Using pip

```bash
# Install
pip install naics-mcp-server

# Initialize database
naics-mcp init

# Generate embeddings
naics-mcp embeddings

# Run the server
naics-mcp-server
```

## Installation

### From PyPI

```bash
pip install naics-mcp-server
```

### From Source

```bash
git clone https://github.com/your-org/naics-mcp-server.git
cd naics-mcp-server
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Configuration

The server is configured via environment variables with sensible defaults.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `NAICS_DATABASE_PATH` | `~/.cache/naics-mcp-server/naics.duckdb` | Path to DuckDB database |
| `NAICS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `NAICS_DEBUG` | `false` | Enable debug mode |

### Search Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `NAICS_HYBRID_WEIGHT_SEMANTIC` | `0.7` | Weight for semantic search (0-1) |
| `NAICS_MIN_CONFIDENCE` | `0.3` | Minimum confidence threshold |
| `NAICS_DEFAULT_LIMIT` | `10` | Default results limit |
| `NAICS_QUERY_TIMEOUT_SECONDS` | `5` | Query timeout |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `NAICS_ENABLE_QUERY_EXPANSION` | `true` | Expand queries with synonyms |
| `NAICS_ENABLE_CROSS_REFERENCES` | `true` | Include cross-reference checks |
| `NAICS_ENABLE_AUDIT_LOG` | `true` | Log search queries for audit |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `NAICS_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `NAICS_LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `NAICS_LOG_FILE` | `None` | Optional file path for logs |

### Example `.env` File

```bash
# Database
NAICS_DATABASE_PATH=/data/naics.duckdb

# Search tuning
NAICS_HYBRID_WEIGHT_SEMANTIC=0.7
NAICS_MIN_CONFIDENCE=0.3

# Logging
NAICS_LOG_LEVEL=INFO
NAICS_LOG_FORMAT=json

# Debug (disable in production)
NAICS_DEBUG=false
```

## Docker

### Using Docker Compose

```bash
# Start production server
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### Development Mode

```bash
# Start with hot reload and debug logging
docker compose --profile dev up naics-dev
```

### Initialize Database

```bash
# Initialize database schema
docker compose --profile init run naics-init

# Generate embeddings (takes a few minutes)
docker compose --profile init run naics-embeddings
```

### Building the Image

```bash
# Build production image
docker build -t naics-mcp-server:latest --target runtime .

# Build development image
docker build -t naics-mcp-server:dev --target development .
```

### Docker Volumes

| Volume | Purpose |
|--------|---------|
| `naics-mcp-data` | Database storage |
| `naics-mcp-cache` | Model cache (sentence-transformers) |
| `naics-mcp-logs` | Application logs |

## Kubernetes

Deploy to Kubernetes using Kustomize:

```bash
# Deploy base configuration
kubectl apply -k k8s/base/

# Deploy production overlay (scaled resources)
kubectl apply -k k8s/overlays/production/
```

The deployment includes:
- Liveness, readiness, and startup probes
- Persistent storage for database and model cache
- ConfigMap-based configuration
- ServiceMonitor for Prometheus Operator

See [k8s/README.md](k8s/README.md) for complete Kubernetes deployment documentation.

## MCP Tools

### Search Tools

| Tool | Description |
|------|-------------|
| `search_naics_codes` | Hybrid semantic/lexical search with confidence scoring |
| `search_index_terms` | Search official 20,398-term NAICS index |
| `find_similar_industries` | Find codes similar to a given code using embeddings |
| `classify_batch` | Classify multiple business descriptions efficiently |

### Hierarchy Tools

| Tool | Description |
|------|-------------|
| `get_code_hierarchy` | Get full ancestor chain (Sector → National Industry) |
| `get_children` | Get immediate children of a code |
| `get_siblings` | Get codes at same level with same parent |

### Classification Tools

| Tool | Description |
|------|-------------|
| `classify_business` | Classify description with detailed reasoning |
| `get_cross_references` | Get exclusions/inclusions for a code |
| `validate_classification` | Validate if a code is correct for a description |

### Analytics Tools

| Tool | Description |
|------|-------------|
| `get_sector_overview` | Summary of sector/subsector structure |
| `compare_codes` | Side-by-side comparison of multiple codes |

### Workbook Tools

| Tool | Description |
|------|-------------|
| `write_to_workbook` | Record classification decisions |
| `search_workbook` | Search past decisions |
| `get_workbook_entry` | Retrieve specific entry |
| `get_workbook_template` | Get form template for structured input |

### Diagnostic Tools

| Tool | Description |
|------|-------------|
| `ping` | Simple liveness check |
| `check_readiness` | Check if server is ready to handle requests |
| `get_server_health` | Detailed health status of all components |
| `get_workflow_guide` | Get recommended classification workflows |

## CLI Usage

```bash
# Run the MCP server
naics-mcp serve

# Search for NAICS codes
naics-mcp search "retail grocery store"
naics-mcp search "dog food manufacturing" --strategy semantic --limit 5

# View code hierarchy
naics-mcp hierarchy 445110

# Show database statistics
naics-mcp stats

# Initialize/rebuild database
naics-mcp init

# Generate/rebuild embeddings
naics-mcp embeddings
naics-mcp embeddings --rebuild  # Force rebuild
```

## Health Checks

The server provides both HTTP endpoints and MCP tools for health monitoring.

### HTTP Endpoints (Port 9090)

| Endpoint | Purpose | Use For |
|----------|---------|---------|
| `GET /health` | Liveness probe | Kubernetes livenessProbe |
| `GET /ready` | Readiness probe | Kubernetes readinessProbe |
| `GET /status` | Detailed status | Monitoring dashboards |
| `GET /metrics` | Prometheus metrics | Prometheus scraping |

```bash
# Check if server is alive
curl http://localhost:9090/health
# {"status": "alive", "timestamp": "2024-01-15T10:30:00Z"}

# Check if ready for traffic
curl http://localhost:9090/ready
# {"status": "ready", "uptime_seconds": 120.5, "timestamp": "..."}

# Get Prometheus metrics
curl http://localhost:9090/metrics
```

See [docs/HTTP_ENDPOINTS.md](docs/HTTP_ENDPOINTS.md) for complete HTTP API documentation.

### MCP Tools

| Tool | Purpose |
|------|---------|
| `ping` | Simple liveness check |
| `check_readiness` | Readiness status |
| `get_server_health` | Detailed component health |

### Health Status Values

| Status | Meaning |
|--------|---------|
| `healthy` | All components ready |
| `degraded` | Some components partial, but operational |
| `unhealthy` | Critical components failed |

## Development

### Setup

```bash
# Clone and install
git clone https://github.com/your-org/naics-mcp-server.git
cd naics-mcp-server
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=naics_mcp_server --cov-report=term-missing

# Lint
ruff check naics_mcp_server/ tests/

# Format
ruff format naics_mcp_server/ tests/

# Type check
mypy naics_mcp_server/
```

### Project Structure

```
naics-mcp-server/
├── naics_mcp_server/
│   ├── __init__.py
│   ├── server.py           # MCP server and tools
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Pydantic configuration
│   ├── core/
│   │   ├── database.py     # DuckDB operations
│   │   ├── embeddings.py   # Sentence transformers
│   │   ├── search_engine.py # Hybrid search
│   │   ├── health.py       # Health checks
│   │   ├── errors.py       # Error handling
│   │   └── validation.py   # Input validation
│   ├── models/
│   │   ├── naics_models.py # NAICS data models
│   │   └── search_models.py # Search result models
│   └── observability/
│       ├── logging.py      # Structured logging
│       └── audit.py        # Audit logging
├── tests/
├── data/                   # Database files (gitignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Architecture

### Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Input Validation│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Expansion │ ─── Synonyms, stemming
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│            Hybrid Search                 │
│  ┌─────────────┐    ┌─────────────┐     │
│  │  Semantic   │    │   Lexical   │     │
│  │  (70%)      │    │   (30%)     │     │
│  └──────┬──────┘    └──────┬──────┘     │
│         └────────┬─────────┘            │
└──────────────────┼──────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ Confidence Scoring│
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Cross-Ref Check  │
         └────────┬────────┘
                  │
                  ▼
            Results
```

### Confidence Scoring

```
overall = (
    0.40 × semantic_score +     # Embedding similarity
    0.20 × lexical_score +      # Term overlap
    0.15 × index_term_match +   # Official index hit
    0.15 × specificity_bonus +  # 6-digit preferred
    0.10 × cross_ref_factor     # Classification guidance
)
```

### Database Schema

```sql
-- NAICS codes (2,125 total across all levels)
naics_nodes (
    node_code VARCHAR PRIMARY KEY,
    level VARCHAR,              -- sector, subsector, etc.
    title VARCHAR,
    description TEXT,
    sector_code, subsector_code, industry_group_code, naics_industry_code,
    raw_embedding_text TEXT
)

-- Vector embeddings (384 dimensions)
naics_embeddings (node_code, embedding FLOAT[384], embedding_text)

-- Official index terms (20,398 entries)
naics_index_terms (term_id, naics_code, index_term, term_normalized)

-- Cross-references for classification guidance
naics_cross_references (
    ref_id, source_code, reference_type,
    reference_text, target_code, excluded_activity
)
```

## Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Search latency (cold) | < 200ms | First search |
| Search latency (warm) | < 50ms | Cached |
| Batch (100 items) | < 10s | Mixed complexity |
| Server startup | < 30s | Including model load |
| Memory usage | < 1GB | Steady state |

## Data Sources

| File | Content |
|------|---------|
| `2-6 digit_2022_Codes.xlsx` | All NAICS codes and titles |
| `2022_NAICS_Descriptions.xlsx` | Full code descriptions |
| `2022_NAICS_Index_File.xlsx` | 20,398 official index terms |
| `2022_NAICS_Cross_References.xlsx` | Classification guidance |

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API_REFERENCE.md) | Complete MCP tools reference |
| [HTTP Endpoints](docs/HTTP_ENDPOINTS.md) | Health check and metrics API |
| [Kubernetes Deployment](k8s/README.md) | Kubernetes deployment guide |
| [Changelog](CHANGELOG.md) | Version history |
| [Production Requirements](docs/PRODUCTION_REQUIREMENTS.md) | Production readiness checklist |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
