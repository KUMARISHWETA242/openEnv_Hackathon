# API Documentation - Satellite Constellation Management

## Canonical Environment

The submission-facing environment is `SatelliteTaskEnv` from `satellite.env`.

```python
from satellite import SatelliteAction, SatelliteTaskEnv

env = SatelliteTaskEnv(task_name="easy")
obs = env.reset()
obs, reward, done, info = env.step(
    SatelliteAction(satellite_actions={0: "capture"})
)
state = env.state()
```

## Core Methods

### `reset() -> SatelliteObservation`

Resets the current task preset and returns the initial observation.

### `step(action: SatelliteAction) -> tuple[SatelliteObservation, SatelliteReward, bool, dict]`

Runs one environment step and returns:
- observation
- typed reward
- done flag
- info dictionary containing reward components, metrics, seed, and task metadata

### `state() -> SatelliteEnvState`

Returns the current typed state snapshot for the active episode.

### `SatelliteTaskEnv.list_tasks() -> dict[str, str]`

Lists the available built-in presets:
- `easy`
- `medium`
- `hard`

## Typed Models

### `SatelliteAction`

Fields:
- `satellite_actions: Dict[int, Literal["capture", "downlink", "maintain", "idle"]]`

### `SatelliteObservation`

Fields:
- `satellites`
- `time_step`
- `ground_stations`
- `weather_conditions`
- `pending_tasks`
- `total_reward`
- `done`
- `reward`
- `metadata`

### `SatelliteReward`

Fields:
- `value`
- `components`

### `SatelliteEnvState`

Fields:
- `episode_id`
- `task_name`
- `step_count`
- `max_steps`
- `seed`
- `done`
- `total_reward`
- `metrics`
- `satellites`
- `ground_stations`
- `weather_conditions`
- `pending_tasks`

## Built-In Tasks

### Easy
- 3 satellites
- 50 steps
- image workload only

### Medium
- 5 satellites
- 100 steps
- mixed image and downlink workload

### Hard
- 8 satellites
- 200 steps
- larger mixed workload with heavier weather pressure

## Deterministic Grading

`TaskGrader` grades a `SatelliteTaskEnv` episode directly:

```python
from satellite import EasyTask, SatelliteTaskEnv, TaskGrader

env = SatelliteTaskEnv(task_name="easy")
env.reset()
score = TaskGrader(EasyTask()).grade_episode(env)
```

The score is deterministic because:
- task seeds are fixed
- task presets are fixed
- grading uses recorded environment metrics rather than parsing free-form traces

## OpenEnv Runtime

The OpenEnv environment root is `satellite/`.

Validate with:

```bash
.venv/bin/openenv validate satellite
```

The FastAPI app entrypoint is:

```text
satellite/server/app.py
```

The manifest is:

```text
satellite/openenv.yaml
```
