FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (non-interactive, minimal extras)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the environment code and useful entry scripts
COPY satellite_env/ ./satellite_env/
COPY openenv.yaml .
COPY run_satellite_rl.py inference.py test_env.py ./

# Make runner scripts executable (optional)
RUN chmod +x run_satellite_rl.py || true

# Expose port if needed (for web interface)
EXPOSE 7860

# Default command
CMD ["python", "-c", "from satellite_env import SatelliteConstellationEnv; print('Environment loaded successfully')"]