# Deployment Guide

This guide covers the two deployment requirements:

1. Deploying the environment as a containerized Hugging Face Space with the `openenv` tag
2. Building and running the environment locally with Docker

## Prerequisites

Make sure Docker is installed and the Docker daemon is running before building.

Check:

```bash
docker --version
docker info
```

If `docker info` fails, start Docker Desktop and wait for the daemon to become available.

## Requirement 1: Hugging Face Spaces Deployment

The `satellite/` directory is the OpenEnv environment root for deployment.

It already includes:
- a Hugging Face Spaces README with `sdk: docker`
- the required `openenv` tag in the README front matter
- an OpenEnv manifest at `satellite/openenv.yaml`
- a root-level Dockerfile at `satellite/Dockerfile`

### Step 1: Use `satellite/` as the Space repository root

The Hugging Face Space should contain the contents of `satellite/` at the repository root.

If you want to test the exact structure locally:

```bash
cd /Users/harrymacbook/Desktop/openEnv_Hackathon/satellite
ls
```

You should see:

```text
Dockerfile
README.md
openenv.yaml
pyproject.toml
server/
```

### Step 2: Confirm the Space metadata

The Hugging Face Space metadata is already present in:

`/Users/harrymacbook/Desktop/openEnv_Hackathon/satellite/README.md`

It includes:

```yaml
---
title: Satellite Environment Server
sdk: docker
app_port: 8000
tags:
  - openenv
---
```

### Step 3: Push `satellite/` to a Docker Space

Create a new Hugging Face Space and choose:
- SDK: `Docker`
- Visibility: your choice

Then push the contents of `satellite/` as the Space repository:

```bash
cd /Users/harrymacbook/Desktop/openEnv_Hackathon/satellite
git init
git add .
git commit -m "Initial OpenEnv satellite Space"
git branch -M main
git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
git push -u origin main
```

### Step 4: Verify the deployed Space

Once the build finishes, the app should be available on port `8000` inside the container and expose the OpenEnv HTTP server.

Expected server entrypoint:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Files involved:
- `/Users/harrymacbook/Desktop/openEnv_Hackathon/satellite/Dockerfile`
- `/Users/harrymacbook/Desktop/openEnv_Hackathon/satellite/README.md`
- `/Users/harrymacbook/Desktop/openEnv_Hackathon/satellite/openenv.yaml`

## Requirement 2: Containerized Execution

The environment can be built and run locally with Docker from the `satellite/` directory.

### Step 1: Build the image

From the project root:

```bash
cd /Users/harrymacbook/Desktop/openEnv_Hackathon
docker build -t satellite-openenv ./satellite
```

Or directly from the environment root:

```bash
cd /Users/harrymacbook/Desktop/openEnv_Hackathon/satellite
docker build -t satellite-openenv .
```

### Step 2: Run the container

```bash
docker run --rm -p 8000:8000 --name satellite-openenv satellite-openenv
```

### Step 3: Verify the server is up

In another terminal:

```bash
curl http://localhost:8000/health
```

You can also inspect the schema:

```bash
curl http://localhost:8000/schema
```

### Step 4: Stop the container

If running in the foreground:

```bash
Ctrl+C
```

If running detached:

```bash
docker stop satellite-openenv
```

## Optional: Run In Detached Mode

```bash
docker run -d -p 8000:8000 --name satellite-openenv satellite-openenv
docker logs -f satellite-openenv
```

## Optional: Validate Before Deployment

From the repository root:

```bash
.venv/bin/openenv validate satellite
```

## Summary

For local Docker:

```bash
cd /Users/harrymacbook/Desktop/openEnv_Hackathon
docker build -t satellite-openenv ./satellite
docker run --rm -p 8000:8000 --name satellite-openenv satellite-openenv
curl http://localhost:8000/health
```

For Hugging Face Spaces:

```bash
cd /Users/harrymacbook/Desktop/openEnv_Hackathon/satellite
git init
git add .
git commit -m "Initial OpenEnv satellite Space"
git branch -M main
git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
git push -u origin main
```
