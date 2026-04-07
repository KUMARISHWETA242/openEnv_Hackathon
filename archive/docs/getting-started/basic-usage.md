# Basic Usage

Learn the fundamental concepts and API usage of the Satellite Constellation Management Environment.

## Core Concepts

### Environment Lifecycle

```python
from satellite_env import SatelliteConstellationEnv, Action

# 1. Create environment
env = SatelliteConstellationEnv(
    num_satellites=5,    # Number of satellites
    max_steps=100        # Maximum steps per episode
)

# 2. Reset for new episode
observation = env.reset()

# 3. Main loop
done = False
while not done:
    # Choose actions
    action = Action(satellite_actions={
        0: "capture",
        1: "maintain",
        2: "downlink",
        3: "idle",
        4: "capture"
    })

    # Execute step
    observation, reward, done, info = env.step(action)

    # Process results
    print(f"Reward: {reward.value}, Done: {done}")

# 4. Get final state
final_state = env.state()
```

### Understanding Observations

The observation contains all information about the current environment state:

```python
observation = env.reset()

# Satellite information
for satellite in observation.satellites:
    print(f"Satellite {satellite.id}:")
    print(f"  Position: {satellite.position}")
    print(f"  Battery: {satellite.battery}%")
    print(f"  Storage: {satellite.storage}%")
    print(f"  Last Action: {satellite.last_action}")

# Environment state
print(f"Time Step: {observation.time_step}")
print(f"Ground Stations: {observation.ground_stations}")
print(f"Weather: {observation.weather_conditions}")
print(f"Pending Tasks: {len(observation.pending_tasks)}")
```

### Action Structure

Actions are specified per satellite:

```python
from satellite_env import Action

# All satellites take the same action
action1 = Action(satellite_actions={
    i: "idle" for i in range(env.num_satellites)
})

# Different actions per satellite
action2 = Action(satellite_actions={
    0: "capture",   # Take image
    1: "maintain",  # Charge battery
    2: "downlink",  # Send data
    3: "idle"       # Do nothing
})

# Partial actions (unspecified satellites default to 'idle')
action3 = Action(satellite_actions={
    0: "capture",
    2: "maintain"
})
```

## Action Types

### Capture Action
```python
# Take an Earth image
action = Action(satellite_actions={satellite_id: "capture"})

# Requirements:
# - Battery > 5%
# - Storage < 90%
# - Good weather conditions (implicit)

# Effects:
# - Battery -= 5
# - Storage += 10
# - Reward += 10
```

### Downlink Action
```python
# Send data to ground station
action = Action(satellite_actions={satellite_id: "downlink"})

# Requirements:
# - Battery > 2%
# - Within range of ground station
# - Storage > 0

# Effects:
# - Battery -= 2
# - Storage -= min(20, current_storage)
# - Reward += 2 × data_sent
```

### Maintain Action
```python
# Perform maintenance/charging
action = Action(satellite_actions={satellite_id: "maintain"})

# Requirements:
# - None (always available)

# Effects:
# - Battery = min(100, battery + 20)
# - Reward += 5
```

### Idle Action
```python
# Do nothing
action = Action(satellite_actions={satellite_id: "idle"})

# Effects:
# - No changes
# - Reward = 0
```

## Reward System

### Understanding Rewards

Rewards provide feedback on agent performance:

```python
observation, reward, done, info = env.step(action)

print(f"Total Reward: {reward.value}")
print("Reward Components:")
for component, value in reward.components.items():
    print(f"  {component}: {value}")
```

### Reward Components

| Action | Success Reward | Penalty | Conditions |
|--------|----------------|---------|------------|
| capture | +10 | -1 | Battery > 5%, Storage < 90% |
| downlink | +2 × data_sent | -1 | In range, Battery > 2% |
| maintain | +5 | 0 | Always available |
| idle | 0 | 0 | Always available |

### Reward Design Principles

- **Partial Progress**: Rewards given for each successful action
- **Penalties**: Negative rewards for invalid actions
- **Scalable**: Rewards scale with task complexity
- **Informative**: Component breakdown shows what worked

## Error Handling

### Invalid Actions

The environment handles invalid actions gracefully:

```python
# This action will be penalized
action = Action(satellite_actions={
    0: "capture"  # If battery too low
})

observation, reward, done, info = env.step(action)
# reward.value will be negative due to penalty
```

### Boundary Conditions

```python
# Battery and storage are clamped to valid ranges
satellite.battery = max(0, min(100, satellite.battery))
satellite.storage = max(0, min(100, satellite.storage))
```

## Task Integration

### Using Built-in Tasks

```python
from satellite_env import EasyTask, MediumTask, HardTask, TaskGrader

# Choose a task
task = EasyTask()
env = SatelliteConstellationEnv()
task.setup_environment(env)

# Create grader
grader = TaskGrader(task)

# Run episode
observation = env.reset()
actions_history = []

# ... run your agent ...

# Evaluate performance
final_state = env.state()
score = grader.grade_episode(env, actions_history, final_state)
print(f"Task Score: {score:.3f}")  # 0.0 to 1.0
```

### Task Criteria

Each task has specific success criteria:

```python
criteria = task.get_success_criteria()
print("Success Criteria:")
for key, value in criteria.items():
    print(f"  {key}: {value}")
```

## Best Practices

### Resource Management
```python
def resource_aware_policy(observation):
    actions = {}
    for satellite in observation.satellites:
        if satellite.battery < 20:
            actions[satellite.id] = "maintain"  # Critical battery
        elif satellite.storage > 90:
            actions[satellite.id] = "downlink"  # Full storage
        elif satellite.battery > 30:
            actions[satellite.id] = "capture"  # Ready for action
        else:
            actions[satellite.id] = "idle"     # Conserve energy
    return Action(satellite_actions=actions)
```

### Communication Awareness
```python
def communication_aware_policy(observation):
    actions = {}
    for satellite in observation.satellites:
        # Check if satellite can downlink
        can_downlink = any(
            ((satellite.position[0] - gs[0])**2 +
             (satellite.position[1] - gs[1])**2) < 500**2
            for gs in observation.ground_stations
        )

        if satellite.storage > 80 and can_downlink:
            actions[satellite.id] = "downlink"
        elif satellite.battery > 25:
            actions[satellite.id] = "capture"
        else:
            actions[satellite.id] = "maintain"
    return Action(satellite_actions=actions)
```

### Weather Considerations
```python
def weather_aware_policy(observation):
    # Check overall weather conditions
    avg_cloud_cover = sum(observation.weather_conditions.values()) / len(observation.weather_conditions)

    actions = {}
    for satellite in observation.satellites:
        if avg_cloud_cover < 0.3 and satellite.battery > 20:
            actions[satellite.id] = "capture"  # Good imaging conditions
        elif satellite.battery < 50:
            actions[satellite.id] = "maintain"
        else:
            actions[satellite.id] = "idle"
    return Action(satellite_actions=actions)
```

---

[← Quick Start](../getting-started/quick-start.md) | [User Guide →](../user-guide/overview.md)