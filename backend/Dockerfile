# Use Python 3.11 runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    motor \
    pydantic \
    python-dotenv \
    aiohttp \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    pillow

# Copy application code
COPY . .

# Expose port
EXPOSE 10000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
>>>>>>> fc3c09125edee633d17968cf9bbef449cabef6f8:Dockerfile

# Copy application code
COPY . .

# Expose port
EXPOSE 10000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]