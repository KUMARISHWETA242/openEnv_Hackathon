# Development Guide - Satellite Constellation Management

## Local Workflow

Create the local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install "openenv-core[core]"
```

Optional project dependencies can be installed from `satellite/pyproject.toml`.

## Important Paths

```text
openEnv_Hackathon/
├── dashboard.py
├── inference.py
├── satellite/
│   ├── constellation.py
│   ├── env.py
│   ├── graders.py
│   ├── models.py
│   ├── openenv.yaml
│   ├── tasks.py
│   └── server/
│       ├── app.py
│       └── satellite_environment.py
└── docs/
```

## Which Class To Extend

- Extend `SatelliteConstellationEnv` when changing low-level simulator dynamics
- Extend `Task` in `satellite/tasks.py` when creating a new preset
- Extend `TaskGrader` in `satellite/graders.py` when scoring needs to change
- Use `SatelliteTaskEnv` when you want the submission-facing typed API

## Creating A New Task

```python
from satellite.tasks import Task

class CustomTask(Task):
    def __init__(self):
        super().__init__("Custom", "Custom mixed operations task")

    def setup_environment(self, env):
        env.num_satellites = 6
        env.max_steps = 120
        env.seed = 404
        env.weather = {"region1": 0.4, "region2": 0.2}
        env.pending_tasks = [
            {"id": "img-1", "type": "image_capture", "region": "region1", "priority": 2},
            {"id": "down-1", "type": "data_downlink", "station": 0, "priority": 2, "units_remaining": 15},
        ]

    def get_success_criteria(self):
        return {
            "min_images_captured": 4,
            "min_data_downlinked": 40,
            "min_tasks_completed": 6,
            "min_battery_final": 25,
            "max_invalid_actions": 6,
            "max_steps": 120,
        }
```

## Grading

The built-in grader consumes a `SatelliteTaskEnv` instance directly:

```python
from satellite import SatelliteTaskEnv, TaskGrader, MediumTask

env = SatelliteTaskEnv(task_name="medium")
env.reset()
score = TaskGrader(MediumTask()).grade_episode(env)
```

If you create a custom task, subclass `TaskGrader` only if the default metric-based scoring is not enough.

## Validation

Run validation from the repo root:

```bash
.venv/bin/openenv validate satellite
```

Or from the environment root:

```bash
cd satellite
../.venv/bin/openenv validate .
```

## Baseline Evaluation

The baseline runner:
- uses the OpenAI Python client
- reads credentials from `HF_TOKEN`
- evaluates `easy`, `medium`, and `hard`
- prints per-task scores and one aggregate score

Run it with:

```bash
export HF_TOKEN="your-token"
export MODEL_NAME="your-model"
python3 inference.py
```

## Local Smoke Test

```python
from satellite import SatelliteAction, SatelliteTaskEnv

env = SatelliteTaskEnv(task_name="easy")
obs = env.reset()
obs, reward, done, info = env.step(
    SatelliteAction(satellite_actions={0: "capture"})
)
state = env.state()
print(obs.time_step, reward.value, done, state.metrics)
```
