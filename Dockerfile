# syntax=docker/dockerfile:1

# NAICS MCP Server Dockerfile
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder - Install dependencies and build wheel
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip and build tools
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy project files
COPY pyproject.toml README.md ./
COPY naics_mcp_server/ ./naics_mcp_server/

# Build app wheel only (no dependencies — torch + deps installed separately in runtime)
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.12-slim AS runtime

# Labels
LABEL org.opencontainers.image.title="NAICS MCP Server"
LABEL org.opencontainers.image.description="Intelligent industry classification service for NAICS 2022"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/mfbaig35r/naics-mcp-server"
LABEL io.modelcontextprotocol.server.name="io.github.mfbaig35r/naics-mcp-server"

# Create non-root user for security
RUN groupadd --gid 1000 naics \
    && useradd --uid 1000 --gid naics --shell /bin/bash --create-home naics

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling ~1.3GB CUDA variant)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy app wheel from builder and install with all remaining deps
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# Create directories for data and cache
RUN mkdir -p /app/data /app/cache /app/logs \
    && chown -R naics:naics /app

# Bake in the pre-built DuckDB database (NAICS codes + embeddings)
COPY --chown=naics:naics data/naics.duckdb /app/data/naics.duckdb

# Pre-download the sentence-transformers model so there's no runtime fetch
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence-transformers \
    HF_HOME=/app/cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && chown -R naics:naics /app/cache

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # NAICS server configuration
    NAICS_DATABASE_PATH=/app/data/naics.duckdb \
    NAICS_LOG_LEVEL=INFO \
    NAICS_LOG_FORMAT=json \
    # HTTP server for health/metrics
    NAICS_HTTP_ENABLED=true \
    NAICS_HTTP_HOST=0.0.0.0 \
    NAICS_HTTP_PORT=9090

# Expose HTTP port for health checks and metrics
EXPOSE 9090

# Switch to non-root user
USER naics

# Health check using HTTP endpoint (faster and cleaner than Python)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Default command - run the MCP server
ENTRYPOINT ["naics-mcp-server"]

# ============================================================================
# Stage 3: Development image with dev dependencies
# ============================================================================
FROM runtime AS development

USER root

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    ruff \
    mypy

# Switch back to non-root user
USER naics

# Override command for development
CMD ["--debug"]
