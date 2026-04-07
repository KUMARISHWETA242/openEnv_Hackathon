---
title: Satellite Environment Server
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Satellite Environment

This is the OpenEnv environment root for the satellite constellation submission.

## What It Exposes

- OpenEnv manifest: `openenv.yaml`
- FastAPI/OpenEnv server: `server/app.py`
- Canonical simulator and typed environment API
- Three built-in task presets: `easy`, `medium`, `hard`

## Validate

From the repository root:

```bash
../.venv/bin/openenv validate .
```

Or from one directory up:

```bash
.venv/bin/openenv validate satellite
```

## Run Locally

```bash
python -m satellite.server.app --port 8000
```

## Python Usage

```python
from satellite import SatelliteAction, SatelliteTaskEnv

env = SatelliteTaskEnv(task_name="medium")
obs = env.reset()
obs, reward, done, info = env.step(
    SatelliteAction(satellite_actions={0: "capture", 1: "maintain"})
)
state = env.state()
```

## Task Progression

| Task | Satellites | Steps | Workload |
|------|------------|-------|----------|
| Easy | 3 | 50 | image tasks |
| Medium | 5 | 100 | mixed image + downlink |
| Hard | 8 | 200 | larger mixed workload with stronger constraints |

## Baseline Evaluation

The repository root contains `inference.py`, which:
- uses the OpenAI Python client by default
- reads credentials from `HF_TOKEN`
- evaluates all three tasks
- reports per-task and aggregate scores

For offline smoke tests:

```bash
BASELINE_POLICY=heuristic python ../inference.py
```
