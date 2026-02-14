# Changelog

All notable changes to the NAICS MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Kubernetes manifests for deployment (`k8s/base/`)
  - Namespace, ConfigMap, PVC, Deployment, Service, ServiceMonitor
  - Kustomize support for easy deployment
  - Production overlay with scaled resources
- HTTP server for health checks and metrics (port 9090)
  - `/health` - Liveness probe endpoint
  - `/ready` - Readiness probe endpoint
  - `/status` - Detailed server status
  - `/metrics` - Prometheus metrics endpoint
- Graceful shutdown with signal handling and request draining
- Token bucket rate limiting for MCP tools
- Prometheus metrics for observability
- Docker containerization with multi-stage builds
- GitHub Actions CI/CD pipelines
- Comprehensive documentation
  - API reference for all MCP tools
  - HTTP endpoints documentation
  - Kubernetes deployment guide

### Changed
- Docker health check now uses HTTP endpoint instead of Python command
- Improved startup time with optimized embedding loading

### Fixed
- Cross-references returning empty results (missing excluded_activity column)
- Workbook reads failing due to column order mismatch
- Classification workbook schema mismatch
- Database path resolution for MCP server

## [0.1.0] - 2024-01-15

### Added
- Initial release of NAICS MCP Server
- Core MCP tools for NAICS code search and classification
  - `search_naics_codes` - Hybrid semantic/lexical search
  - `search_index_terms` - Search 20,398 official index terms
  - `find_similar_industries` - Semantic similarity search
  - `classify_batch` - Batch classification
  - `classify_business` - Detailed classification with reasoning
  - `get_cross_references` - Exclusion/inclusion checks
  - `validate_classification` - Validate code for description
- Hierarchy navigation tools
  - `get_code_hierarchy` - Full ancestor chain
  - `get_children` - Immediate children
  - `get_siblings` - Same-level alternatives
- Analytics tools
  - `get_sector_overview` - Sector/subsector summary
  - `compare_codes` - Side-by-side comparison
- Classification workbook for tracking decisions
  - `write_to_workbook` - Record classification
  - `search_workbook` - Search past decisions
  - `get_workbook_entry` - Retrieve entry
  - `get_workbook_template` - Get form templates
- Diagnostic tools
  - `ping` - Liveness check
  - `check_readiness` - Readiness check
  - `get_server_health` - Detailed health status
  - `get_metrics` - Prometheus metrics
  - `get_workflow_guide` - Classification workflows
- DuckDB database with NAICS 2022 data
  - 2,125 NAICS codes across 5 levels
  - 20,398 official index terms
  - 4,500+ cross-references
- Sentence transformer embeddings (all-MiniLM-L6-v2)
- Hybrid search with configurable semantic/lexical weights
- Confidence scoring with detailed breakdowns
- Query expansion for improved recall
- Structured logging with JSON format
- Pydantic-based configuration management
- Comprehensive input validation
- Error handling with retry logic and graceful degradation
- marimo ETL pipeline for data ingestion

### Dependencies
- Python 3.11+
- MCP (Model Context Protocol)
- DuckDB
- sentence-transformers
- Pydantic
- Starlette/Uvicorn (HTTP server)
- prometheus-client

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-01-15 | Initial release with core classification tools |

[Unreleased]: https://github.com/your-org/naics-mcp-server/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/naics-mcp-server/releases/tag/v0.1.0
