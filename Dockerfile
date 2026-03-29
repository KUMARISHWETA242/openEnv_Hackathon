FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the environment code
COPY satellite_env/ ./satellite_env/
COPY openenv.yaml .
COPY baseline.py .

# Make baseline script executable
RUN chmod +x baseline.py

# Expose port if needed (for web interface)
EXPOSE 7860

# Default command
CMD ["python", "-c", "from satellite_env import SatelliteConstellationEnv; print('Environment loaded successfully')"]