# AI Forge Dockerfile
# For testing and non-GPU environments
# Note: Production training should run natively on Mac for Apple Silicon optimization

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/output /app/logs /app/cache

# Environment variables
ENV PYTHONPATH=/app
ENV AI_FORGE_DATA_DIR=/app/data
ENV AI_FORGE_OUTPUT_DIR=/app/output
ENV AI_FORGE_LOGS_DIR=/app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the API server
CMD ["uvicorn", "ai_forge.conductor.service:app", "--host", "0.0.0.0", "--port", "8000"]

# For development, you can override with:
# docker run -it ai-forge python -m pytest
# docker run -it ai-forge python -c "from ai_forge.training import TrainingForge; print('OK')"
