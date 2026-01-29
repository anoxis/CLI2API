# Multi-stage build for CLI2API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN useradd -m -u 1000 cli2api && \
    mkdir -p /app && \
    chown -R cli2api:cli2api /app

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY --chown=cli2api:cli2api pyproject.toml ./

# Install Python dependencies
RUN pip install -e .

# Copy application code
COPY --chown=cli2api:cli2api cli2api/ ./cli2api/

# Switch to non-root user
USER cli2api

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "cli2api.main:app", "--host", "0.0.0.0", "--port", "8000"]
