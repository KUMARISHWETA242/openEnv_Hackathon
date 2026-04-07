# Satellite Constellation Management Environment - API Documentation

## Overview

The Satellite Constellation Management Environment implements the OpenEnv specification for reinforcement learning. This document provides detailed API reference for developers.

## Core Classes

### SatelliteConstellationEnv

The main environment class implementing the OpenEnv interface.

#### Constructor
```python
SatelliteConstellationEnv(num_satellites: int = 5, max_steps: int = 100)
```

**Parameters:**
- `num_satellites` (int): Number of satellites in the constellation (default: 5)
- `max_steps` (int): Maximum steps per episode (default: 100)

#### Methods

##### reset() -> Observation
Resets the environment to initial state.

**Returns:** Initial observation

##### step(action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]
Executes one time step in the environment.

**Parameters:**
- `action` (Action): Action to execute

**Returns:**
- `observation` (Observation): New observation after action
- `reward` (Reward): Reward received
- `done` (bool): Whether episode is complete
- `info` (Dict): Additional information

##### state() -> Dict[str, Any]
Returns the current internal state of the environment.

**Returns:** Dictionary containing full environment state

## Data Models

### Observation
Pydantic model representing the environment observation.

**Fields:**
- `satellites` (List[SatelliteState]): States of all satellites
- `time_step` (int): Current time step
- `ground_stations` (List[Tuple[float, float]]): Ground station locations (lat, lon)
- `weather_conditions` (Dict[str, float]): Weather conditions by region (0-1 cloud cover)
- `pending_tasks` (List[Dict[str, Any]]): Currently pending tasks

### SatelliteState
State of an individual satellite.

**Fields:**
- `id` (int): Unique satellite identifier
- `position` (Tuple[float, float, float]): 3D position (x, y, z) in km
- `battery` (float): Battery level (0-100%)
- `storage` (float): Storage usage (0-100%)
- `last_action` (str): Last action performed

### Action
Pydantic model representing agent actions.

**Fields:**
- `satellite_actions` (Dict[int, str]): Mapping of satellite ID to action

**Valid Actions:**
- `"capture"`: Capture an Earth image
- `"downlink"`: Transmit data to ground station
- `"maintain"`: Perform maintenance operations
- `"idle"`: No operation

### Reward
Pydantic model representing reward information.

**Fields:**
- `value` (float): Total reward value
- `components` (Dict[str, float]): Breakdown of reward components

## Task Classes

### Task (Abstract Base Class)
Base class for all tasks.

#### Methods
- `setup_environment(env: SatelliteConstellationEnv)`: Configure environment for task
- `get_success_criteria() -> Dict[str, Any]`: Return success criteria

### EasyTask
Basic imaging task with 3 satellites.

**Configuration:**
- 3 satellites
- 50 maximum steps
- Focus on image capture and basic resource management

### MediumTask
Data management task with 5 satellites.

**Configuration:**
- 5 satellites
- 100 maximum steps
- Includes data downlink requirements

### HardTask
Full constellation coordination with 8 satellites.

**Configuration:**
- 8 satellites
- 200 maximum steps
- Complex weather and communication constraints

## Grader Classes

### TaskGrader
Evaluates agent performance on tasks.

#### Constructor
```python
TaskGrader(task: Task)
```

#### Methods
- `grade_episode(env, actions, final_state) -> float`: Returns score 0.0-1.0

## Environment Dynamics

### Satellite Movement
Satellites follow simplified orbital mechanics:
- Position updated with random perturbations each step
- Approximate low Earth orbit characteristics

### Resource Management
- **Battery**: Drains 0.5% per step, actions consume additional power
- **Storage**: Fills with image captures, empties with data downlink
- **Communication**: Limited to proximity with ground stations

### Action Effects

| Action | Battery Cost | Storage Effect | Reward |
|--------|-------------|----------------|---------|
| capture | -5% | +10% | +10 |
| downlink | -2% | -20% | +2 × data sent |
| maintain | +20% | 0 | +5 |
| idle | 0 | 0 | 0 |

### Reward Components
- **Task Completion**: Points for successful captures/downlinks
- **Resource Management**: Bonuses for efficient resource use
- **Penalties**: Negative rewards for invalid actions

## Configuration Files

### openenv.yaml
Environment metadata following OpenEnv specification.

**Required Fields:**
- `name`: Environment identifier
- `version`: Version string
- `description`: Human-readable description
- `observation_space`: Observation space description
- `action_space`: Action space description
- `reward_space`: Reward space description
- `episode_max_length`: Maximum episode length
- `tasks`: List of available tasks

## Usage Examples

### Basic Environment Interaction
```python
from satellite_env import SatelliteConstellationEnv, Action

env = SatelliteConstellationEnv(num_satellites=3)
obs = env.reset()

action = Action(satellite_actions={0: "capture", 1: "maintain", 2: "idle"})
obs, reward, done, info = env.step(action)
```

### Running Tasks
```python
from satellite_env import EasyTask, TaskGrader

task = EasyTask()
env = SatelliteConstellationEnv()
task.setup_environment(env)

grader = TaskGrader(task)
# ... run episode ...
score = grader.grade_episode(env, actions, final_state)
```

## Extension Points

### Custom Tasks
Create new tasks by inheriting from `Task`:

```python
class CustomTask(Task):
    def setup_environment(self, env):
        # Configure environment
        pass

    def get_success_criteria(self):
        return {"custom_metric": threshold}
```

### Environment Modifications
Extend `SatelliteConstellationEnv` to add:
- More realistic orbital mechanics
- Additional satellite capabilities
- Complex weather models
- Communication protocols

## Error Handling

The environment includes robust error handling:
- Invalid actions default to 'idle'
- Resource constraints prevent impossible operations
- Graceful degradation on boundary conditions

## Performance Considerations

- Environment is lightweight and runs efficiently
- Scales to 100+ satellites for large-scale simulations
- Minimal dependencies for easy deployment