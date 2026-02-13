# NAICS MCP Server - Production Readiness Requirements

## Overview

This document outlines the requirements for making the NAICS MCP Server production-ready. The server provides NAICS 2022 classification services via the Model Context Protocol (MCP).

**Current State**: Functional prototype with core features working
**Target State**: Production-ready service suitable for enterprise deployment

---

## 1. Testing Requirements

### 1.1 Unit Tests

| Component | Coverage Target | Priority |
|-----------|-----------------|----------|
| `NAICSDatabase` | 90% | P0 |
| `NAICSSearchEngine` | 90% | P0 |
| `TextEmbedder` | 80% | P1 |
| `ClassificationWorkbook` | 85% | P1 |
| Models (`NAICSCode`, `ConfidenceScore`, etc.) | 95% | P0 |

**Test cases must include:**
- Happy path for all public methods
- Edge cases (empty inputs, special characters, Unicode)
- Error conditions (database unavailable, invalid codes)
- Boundary conditions (max length descriptions, batch limits)

### 1.2 Integration Tests

| Test Scenario | Description |
|---------------|-------------|
| Full classification flow | Description → search → cross-ref check → result |
| Workbook persistence | Write → read → search → verify |
| Batch classification | Process 100+ items, verify accuracy |
| Server lifecycle | Startup → serve requests → graceful shutdown |

### 1.3 Accuracy Tests

| Metric | Target | Measurement |
|--------|--------|-------------|
| Top-1 accuracy | ≥75% | Correct 6-digit code in first result |
| Top-3 accuracy | ≥90% | Correct code in top 3 results |
| Cross-ref detection | ≥95% | Exclusions correctly identified |

**Test dataset**: 200+ business descriptions with known NAICS codes (manually validated).

### 1.4 Performance Tests

| Metric | Target | Condition |
|--------|--------|-----------|
| Single search latency | <200ms p95 | Cold cache |
| Single search latency | <50ms p95 | Warm cache |
| Batch classification | <5s for 100 items | Mixed complexity |
| Memory usage | <500MB | Steady state with embeddings loaded |
| Startup time | <10s | Including embedding initialization |

---

## 2. Logging Requirements

### 2.1 Log Levels

| Level | Usage |
|-------|-------|
| `ERROR` | Unrecoverable errors, exceptions that affect functionality |
| `WARNING` | Recoverable issues, degraded performance, fallback behavior |
| `INFO` | Significant events: startup, shutdown, configuration changes |
| `DEBUG` | Detailed operational info: search parameters, cache hits |
| `TRACE` | Very detailed: embedding vectors, SQL queries (dev only) |

### 2.2 Structured Logging Format

All logs must be JSON-formatted for production:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "naics_mcp_server.search_engine",
  "message": "Search completed",
  "context": {
    "request_id": "req_abc123",
    "session_id": "sess_xyz789",
    "tool": "classify_business"
  },
  "data": {
    "query_length": 45,
    "strategy": "hybrid",
    "results_count": 5,
    "top_confidence": 0.85,
    "latency_ms": 127,
    "cache_hit": false
  }
}
```

### 2.3 Required Log Events

#### Startup/Shutdown
- Server start with configuration summary
- Database connection established/failed
- Embedding model loaded (with load time)
- Embeddings initialized (count, time)
- Server ready to accept requests
- Graceful shutdown initiated/completed

#### Request Processing
- Request received (tool name, truncated input)
- Search executed (strategy, results count, latency)
- Classification completed (top code, confidence)
- Cross-reference check (warnings found/not found)
- Workbook write/read operations
- Request completed (total latency)

#### Errors
- Database connection failures (with retry count)
- Embedding generation failures
- Invalid input received (sanitized)
- Timeout exceeded
- Unexpected exceptions (with stack trace)

#### Performance
- Slow queries (>200ms) with query details
- Cache evictions
- Memory warnings (if approaching limits)
- Batch processing progress (every 10%)

### 2.4 Log Configuration

```python
# Environment variables
NAICS_LOG_LEVEL=INFO          # Minimum log level
NAICS_LOG_FORMAT=json         # json or text
NAICS_LOG_FILE=/var/log/naics/server.log  # Optional file output
NAICS_LOG_MAX_SIZE=100MB      # Rotation size
NAICS_LOG_RETENTION=7         # Days to retain
```

### 2.5 Sensitive Data Handling

**Must NOT log:**
- Full business descriptions (truncate to 100 chars)
- Internal database paths in production
- Stack traces for client-facing errors (log internally only)
- Any PII if present in descriptions

**Must sanitize:**
- Replace numbers >6 digits with `[REDACTED]`
- Truncate long strings with `...`

---

## 3. Error Handling Requirements

### 3.1 Error Categories

| Category | HTTP-like Code | Behavior |
|----------|----------------|----------|
| Input validation | 400 | Return clear error message |
| Not found | 404 | Return empty result with message |
| Server error | 500 | Log full details, return generic message |
| Timeout | 504 | Return partial results if available |
| Rate limited | 429 | Return retry-after guidance |

### 3.2 Graceful Degradation

| Failure | Degraded Behavior |
|---------|-------------------|
| Embeddings not ready | Fall back to lexical search (current) |
| Cross-ref table missing | Skip cross-ref check, warn in response |
| Workbook unavailable | Return results without persistence |
| Slow database | Return cached results if available |

### 3.3 Retry Logic

| Operation | Max Retries | Backoff |
|-----------|-------------|---------|
| Database connection | 3 | Exponential (1s, 2s, 4s) |
| Embedding generation | 2 | Linear (5s) |
| External API calls | 3 | Exponential with jitter |

---

## 4. Input Validation Requirements

### 4.1 Business Description

| Constraint | Value | Action |
|------------|-------|--------|
| Min length | 10 chars | Reject with message |
| Max length | 5,000 chars | Truncate with warning |
| Encoding | UTF-8 | Normalize or reject |
| Empty/whitespace | - | Reject with message |

### 4.2 NAICS Code

| Constraint | Value | Action |
|------------|-------|--------|
| Format | 2-6 digits | Reject if invalid |
| Existence | Must be in database | Return not found |
| Leading zeros | Preserve | Handle "01" vs "1" |

### 4.3 Batch Operations

| Constraint | Value | Action |
|------------|-------|--------|
| Max batch size | 100 items | Reject if exceeded |
| Individual item limits | Same as single | Apply per-item |
| Timeout | 60 seconds total | Return partial results |

---

## 5. Packaging & Distribution Requirements

### 5.1 PyPI Package

- Package name: `naics-mcp-server`
- Versioning: Semantic versioning (MAJOR.MINOR.PATCH)
- Dependencies: Pinned with compatible ranges
- Python support: 3.10, 3.11, 3.12

### 5.2 Installation Options

```bash
# Option 1: PyPI (requires ETL for data)
pip install naics-mcp-server
naics-mcp-server init  # Downloads/builds database

# Option 2: With bundled data
pip install naics-mcp-server[data]

# Option 3: Docker
docker run -p 8080:8080 naics-mcp-server:latest
```

### 5.3 Docker Image

- Base image: `python:3.12-slim`
- Multi-stage build (builder + runtime)
- Non-root user
- Health check endpoint
- Configurable via environment variables
- Size target: <500MB

### 5.4 Required Files

```
naics-mcp-server/
├── pyproject.toml        # Package metadata, dependencies
├── README.md             # Quickstart, examples
├── LICENSE               # License file
├── CHANGELOG.md          # Version history
├── Dockerfile            # Container build
├── docker-compose.yml    # Local development
└── .github/
    └── workflows/
        ├── test.yml      # CI tests
        ├── release.yml   # PyPI publish
        └── docker.yml    # Image build
```

---

## 6. Documentation Requirements

### 6.1 User Documentation

| Document | Contents |
|----------|----------|
| README.md | Overview, quickstart, basic examples |
| INSTALL.md | Detailed installation for all platforms |
| USAGE.md | All tools with examples |
| TROUBLESHOOTING.md | Common issues and solutions |

### 6.2 API Reference

- Auto-generated from docstrings
- All MCP tools documented with:
  - Description
  - Parameters (types, constraints)
  - Return values
  - Example requests/responses
  - Error conditions

### 6.3 Developer Documentation

| Document | Contents |
|----------|----------|
| ARCHITECTURE.md | System design, data flow |
| CONTRIBUTING.md | Dev setup, coding standards |
| DATA_PIPELINE.md | ETL process, data updates |
| DEPLOYMENT.md | Production deployment guide |

---

## 7. Observability Requirements

### 7.1 Health Check Endpoint

The `get_server_health` tool must return:

```json
{
  "status": "healthy|degraded|unhealthy",
  "components": {
    "database": {"status": "ready", "latency_ms": 5},
    "embeddings": {"status": "ready", "count": 2125},
    "workbook": {"status": "ready"}
  },
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

### 7.2 Metrics (Future)

| Metric | Type | Labels |
|--------|------|--------|
| `naics_requests_total` | Counter | tool, status |
| `naics_request_duration_seconds` | Histogram | tool |
| `naics_search_results_count` | Histogram | strategy |
| `naics_cache_hits_total` | Counter | cache_type |
| `naics_errors_total` | Counter | error_type |

### 7.3 Audit Log

For compliance, maintain searchable audit log:

| Field | Description |
|-------|-------------|
| timestamp | ISO 8601 |
| session_id | MCP session identifier |
| tool | Tool name invoked |
| input_hash | SHA256 of input (not raw input) |
| result_codes | NAICS codes returned |
| confidence | Top confidence score |
| latency_ms | Processing time |

---

## 8. Data Management Requirements

### 8.1 Database Migrations

- Use versioned migration files (not ad-hoc ALTER TABLE)
- Track applied migrations in metadata table
- Support rollback for failed migrations
- Test migrations against production-like data

### 8.2 NAICS Updates

NAICS is revised periodically (2017 → 2022 → 2027):

- Document update process
- Support multiple NAICS versions (future)
- Provide crosswalk tool for version migration

### 8.3 Backup/Restore

- Document backup procedure
- Test restore process
- Embeddings can be regenerated (document time requirement)

---

## 9. Security Requirements

### 9.1 Input Sanitization

- Escape special characters in SQL (parameterized queries - already done)
- Validate all user inputs before processing
- Reject malformed requests early

### 9.2 Dependencies

- No known vulnerabilities (check with `pip-audit`)
- Pin dependency versions
- Regular security updates

### 9.3 Deployment

- Run as non-root user
- Minimal container permissions
- No secrets in code or logs

---

## 10. CI/CD Requirements

### 10.1 Continuous Integration

On every PR:
- Run unit tests
- Run integration tests
- Check code formatting (ruff)
- Check type hints (mypy)
- Check for security vulnerabilities
- Build package

### 10.2 Continuous Deployment

On release tag:
- Run full test suite
- Build and publish to PyPI
- Build and publish Docker image
- Update documentation

---

## Implementation Priority

| Phase | Items | Duration |
|-------|-------|----------|
| **Phase 1: Foundation** | Testing, Logging, Error Handling | 3-4 days |
| **Phase 2: Polish** | Input Validation, Documentation | 2 days |
| **Phase 3: Distribution** | Packaging, Docker, CI/CD | 2 days |
| **Phase 4: Observability** | Metrics, Audit Log enhancements | 1-2 days |

**Total estimated effort**: 8-10 days

---

## Acceptance Criteria

The server is production-ready when:

1. ✅ All P0 tests pass with >90% coverage
2. ✅ No ERROR-level logs during normal operation
3. ✅ Search latency <200ms p95
4. ✅ Graceful degradation for all failure modes
5. ✅ Package installable via `pip install`
6. ✅ Docker image builds and runs successfully
7. ✅ Documentation complete for all tools
8. ✅ CI pipeline green on main branch
