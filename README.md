# Satellite Constellation Management Environment

This repository contains a real-world OpenEnv submission for satellite fleet operations. Agents must manage image capture, data downlink, maintenance timing, and resource risk across three deterministic task presets.

## Overview And Motivation

This environment models a realistic operations problem: coordinating a satellite fleet that must capture imagery, preserve onboard resources, and downlink data under changing weather and workload pressure.

It is intended as a meaningful agent benchmark because good performance requires:
- balancing short-term task completion against long-term battery and storage health
- choosing among competing operational priorities
- avoiding wasteful or destructive actions across long trajectories
- adapting strategy as task mix and constellation size increase from easy to hard

## What Is Included

- Canonical environment package: `satellite/`
- Typed Pydantic models for observation, action, reward, and state
- Three built-in tasks: `easy`, `medium`, `hard`
- Deterministic task grader returning scores from `0.0` to `1.0`
- Reward shaping for progress, efficiency, and bad behavior penalties
- Baseline inference runner at `inference.py`
- Local validator-ready OpenEnv app manifest at `satellite/openenv.yaml`

## Task Progression

| Task | Satellites | Max Steps | Workload | Main Difficulty |
|------|------------|-----------|----------|-----------------|
| Easy | 3 | 50 | 5 image tasks | Basic resource management |
| Medium | 5 | 100 | 12 mixed tasks | Capture/downlink balancing |
| Hard | 8 | 200 | 20 mixed tasks | Heavier weather, more coordination, stricter grading |

The progression is explicit in both configuration and grading:
- `easy` focuses on simple image completion and battery health
- `medium` adds meaningful downlink workload and more task completions
- `hard` increases fleet size, task count, cloud pressure, and invalid-action sensitivity

## Canonical API

Use `SatelliteTaskEnv` for the submission-facing environment API:

```python
from satellite import SatelliteAction, SatelliteTaskEnv

env = SatelliteTaskEnv(task_name="medium")
observation = env.reset()

observation, reward, done, info = env.step(
    SatelliteAction(satellite_actions={0: "capture", 1: "maintain"})
)

state = env.state()
```

Key methods:
- `reset() -> SatelliteObservation`
- `step(action) -> (SatelliteObservation, SatelliteReward, done, info)`
- `state() -> SatelliteEnvState`
- `SatelliteTaskEnv.list_tasks() -> Dict[str, str]`

## Action And Observation Spaces

### Action Space

The action space is a typed `SatelliteAction` object with one command per satellite:

```python
SatelliteAction(
    satellite_actions={
        0: "capture",
        1: "downlink",
        2: "maintain",
        3: "idle",
    }
)
```

Allowed actions:
- `capture`: collect imagery for an image task
- `downlink`: transmit stored data toward a downlink task
- `maintain`: recover battery and preserve fleet health
- `idle`: take no productive action this step

### Observation Space

The observation space is a typed `SatelliteObservation` object containing:
- `satellites`: per-satellite state with `id`, `position`, `battery`, `storage`, and `last_action`
- `time_step`: current step in the episode
- `ground_stations`: available ground-station coordinates
- `weather_conditions`: cloud cover by region
- `pending_tasks`: currently visible image/downlink tasks
- `total_reward`: cumulative reward so far
- `reward`: immediate reward from the latest step
- `done`: whether the episode has ended
- `metadata`: step metadata such as reward components and metrics

## Reward Model

Rewards are shaped during the trajectory, not only at the end:
- positive reward for completing image and downlink tasks
- additional reward for finishing full downlink workloads
- moderate reward for timely maintenance
- penalties for invalid actions
- penalties for repeated wasteful actions
- penalties for risky low-battery or overfull-storage behavior
- mild penalty for unproductive idling when useful work is available

## Grading

`TaskGrader` scores episodes deterministically from environment metrics, including:
- completed image tasks
- downlinked units
- total tasks completed
- final average battery
- invalid-action rate

## Local Setup

Create the local virtualenv and install the OpenEnv runtime:

```bash
python3 -m venv .venv
.venv/bin/pip install "openenv-core[core]"
```

## Validate The Environment

The OpenEnv environment root is `satellite/`, not the repo root.

Use either:

```bash
.venv/bin/openenv validate satellite
```

or:

```bash
cd satellite
../.venv/bin/openenv validate .
```

## Dashboard

Run the local dashboard with:

```bash
python3 dashboard.py
```

It supports:
- task switching across `easy`, `medium`, and `hard`
- heuristic, random, and manual action selection
- reward and fleet-state inspection

## Baseline Inference

The baseline script evaluates all three tasks and prints:
- per-task score
- per-task reward
- per-task step count
- final aggregate score

Set:

```bash
export HF_TOKEN="your-token"
export MODEL_NAME="your-model"
export API_BASE_URL="https://router.huggingface.co/v1"
python3 inference.py
```

The script uses the OpenAI Python client and reads credentials from `HF_TOKEN`.

### Reproducible Baseline Scores

The repository also supports a deterministic heuristic baseline that can be reproduced locally without a remote model:

```bash
BASELINE_POLICY=heuristic python3 inference.py
```

Current baseline scores:

| Task | Score | Reward | Steps | Done |
|------|-------|--------|-------|------|
| easy | 1.0000 | 34.60 | 50 | True |
| medium | 1.0000 | 126.70 | 100 | True |
| hard | 1.0000 | 170.40 | 200 | True |

Aggregate heuristic baseline score: `1.0000`

## Project Structure

```text
openEnv_Hackathon/
├── dashboard.py
├── inference.py
├── satellite/
│   ├── __init__.py
│   ├── constellation.py
│   ├── env.py
│   ├── graders.py
│   ├── models.py
│   ├── openenv.yaml
│   ├── pyproject.toml
│   ├── tasks.py
│   └── server/
│       ├── app.py
│       └── satellite_environment.py
└── docs/
```
