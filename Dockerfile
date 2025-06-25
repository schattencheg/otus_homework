FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MATPLOTLIB_BACKEND=Agg

# Default command (can be overridden)
CMD ["python", "run_me_hw7.py", "live"]
