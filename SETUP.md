# NAICS MCP Server — Setup Guide

An MCP server for NAICS 2022 industry classification with semantic search, hierarchy navigation, cross-references, and a classification workbook. Backed by DuckDB with 384-dim sentence embeddings.

---

## Quick Start (Docker)

Pull and run the pre-built image:

```bash
docker pull fbaig4/naics-mcp-server
docker run -p 9090:9090 fbaig4/naics-mcp-server
```

The server starts on stdio (MCP transport) with health endpoints on port 9090.

---

## Docker Compose (Recommended)

```bash
git clone https://github.com/mfbaig35r/naics-mcp-server.git
cd naics-mcp-server
```

**Start the server:**

```bash
docker compose up           # foreground
docker compose up -d        # background
```

**Initialize the database (first time only):**

```bash
docker compose --profile init run naics-init
docker compose --profile init run naics-embeddings
```

**Development mode (hot reload):**

```bash
docker compose --profile dev up naics-dev
```

---

## Local Install (pip)

Requires Python 3.11+.

```bash
pip install naics-mcp-server
```

**Initialize and run:**

```bash
naics-mcp init              # Create DuckDB database
naics-mcp embeddings        # Generate embeddings (one-time, takes a few minutes)
naics-mcp serve             # Start the MCP server
```

---

## CLI Reference

```bash
naics-mcp serve                                      # Run MCP server
naics-mcp init                                       # Initialize database
naics-mcp embeddings                                 # Generate embeddings
naics-mcp embeddings --rebuild                       # Force rebuild embeddings
naics-mcp search "retail grocery store"              # Test search
naics-mcp search "dog food manufacturing" --limit 5  # Search with options
naics-mcp hierarchy 445110                           # View code hierarchy
naics-mcp stats                                      # Database statistics
naics-mcp classify-batch suppliers.csv               # Batch classify from CSV
naics-mcp classify-batch in.csv --column name        # Specify description column
naics-mcp classify-batch in.csv --top-n 3            # Top 3 matches per row
naics-mcp classify-batch in.csv -o results.csv       # Custom output path
```

---

## Connect to Claude Code

Add this to your Claude Code MCP settings (`~/.claude/settings.json` or project `.mcp.json`):

**Docker (stdio transport):**

```json
{
  "mcpServers": {
    "naics": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "fbaig4/naics-mcp-server"]
    }
  }
}
```

**Local install (stdio transport):**

```json
{
  "mcpServers": {
    "naics": {
      "command": "naics-mcp",
      "args": ["serve"]
    }
  }
}
```

---

## Environment Variables

All prefixed with `NAICS_`. Defaults work out of the box.

| Variable | Default | Description |
|----------|---------|-------------|
| `NAICS_DATABASE_PATH` | `~/.cache/naics-mcp-server/naics.duckdb` | DuckDB database location |
| `NAICS_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `NAICS_HYBRID_WEIGHT_SEMANTIC` | `0.7` | Semantic vs lexical weight (0-1) |
| `NAICS_MIN_CONFIDENCE` | `0.3` | Minimum confidence threshold |
| `NAICS_DEBUG` | `false` | Enable debug mode |
| `NAICS_LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR |
| `NAICS_LOG_FORMAT` | `json` | `json` or `text` |
| `NAICS_HTTP_ENABLED` | `true` | Enable health/metrics HTTP server |
| `NAICS_HTTP_HOST` | `0.0.0.0` | HTTP bind address |
| `NAICS_HTTP_PORT` | `9090` | HTTP port |
| `NAICS_ENABLE_RATE_LIMITING` | `false` | Enable per-tool rate limiting |
| `NAICS_DEFAULT_RPM` | — | Default requests per minute |
| `NAICS_SEARCH_RPM` | — | Search tool RPM limit |
| `NAICS_CLASSIFY_RPM` | — | Classify tool RPM limit |

Pass to Docker with `-e`:

```bash
docker run -p 9090:9090 -e NAICS_LOG_LEVEL=DEBUG fbaig4/naics-mcp-server
```

---

## Available MCP Tools

### Search
- **search_naics_codes** — Hybrid semantic/lexical search across all NAICS codes
- **search_index_terms** — Search 20,398 official NAICS index terms
- **find_similar_industries** — Find codes similar to a given NAICS code

### Classification
- **classify_business** — Classify a business description with reasoning
- **classify_batch** — Classify multiple descriptions in one call
- **get_cross_references** — Exclusions/inclusions for a code
- **validate_classification** — Check if a code fits a description

### Hierarchy
- **get_code_hierarchy** — Full ancestor chain (Sector → National Industry)
- **get_children** — Immediate children of a code
- **get_siblings** — Codes at the same level under the same parent

### Relationships
- **get_similar_codes** — Semantically similar codes
- **get_cross_sector_alternatives** — Related codes in other sectors
- **get_relationship_stats** — Relationship statistics

### Analytics
- **get_sector_overview** — Sector/subsector structure summary
- **compare_codes** — Side-by-side comparison of multiple codes

### Workbook
- **write_to_workbook** — Record a classification decision
- **search_workbook** — Search past decisions
- **get_workbook_entry** — Retrieve a specific entry
- **get_workbook_template** — Get structured input template

### Diagnostics
- **ping** — Liveness check
- **check_readiness** — Readiness probe
- **get_server_health** — Detailed health status
- **get_metrics** — Prometheus-format metrics

---

## Health Endpoints

When `NAICS_HTTP_ENABLED=true` (default), an HTTP server runs on port 9090:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe (Kubernetes) |
| `GET /ready` | Readiness probe |
| `GET /status` | Detailed server status |
| `GET /metrics` | Prometheus metrics |

---

## Data

The database contains:
- **2,125 NAICS codes** across 5 hierarchy levels (Sector → National Industry)
- **384-dimensional embeddings** per code (all-MiniLM-L6-v2)
- **20,398 official index terms**
- **Cross-reference data** (exclusions/inclusions)

The Docker image ships with the database pre-built. For local installs, run `naics-mcp init` and `naics-mcp embeddings`.

---

## Troubleshooting

**Server won't start:** Run `naics-mcp stats` to verify the database exists and has data. If not, re-run `naics-mcp init && naics-mcp embeddings`.

**Slow first search:** The sentence transformer model loads on first query. Subsequent searches are fast.

**Port conflict on 9090:** Set `NAICS_HTTP_PORT` to another port, or disable with `NAICS_HTTP_ENABLED=false`.

**Docker permission issues:** The container runs as non-root user `naics` (UID 1000). Mount volumes with matching permissions if needed.
