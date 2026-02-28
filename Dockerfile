FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ /app/src/
COPY artifacts/best_model.keras /app/artifacts/best_model.keras

# Expose port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Command to run the application (CPU only inference)
ENV CUDA_VISIBLE_DEVICES="-1"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
