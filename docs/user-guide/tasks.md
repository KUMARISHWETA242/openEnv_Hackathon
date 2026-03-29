# Tasks

Complete description of the three task difficulties, objectives, and evaluation criteria.

## Task Overview

The satellite environment provides three difficulty levels:

- **Easy**: Basic resource management with simple objectives
- **Medium**: Multi-objective optimization with resource constraints
- **Hard**: Complex coordination requiring strategic planning

Each task defines:
- Number of satellites in the constellation
- Maximum episode length
- Success criteria for completion
- Environmental conditions

## Easy Task

### Configuration

```python
class EasyTask(Task):
    name = "Easy"
    num_satellites = 3
    max_steps = 100
    success_criteria = {
        "min_images_captured": 50,
        "min_avg_battery": 0.3,  # 30%
        "max_satellite_failures": 0,
    }
```

### Objectives

**Primary Goal**: Capture at least 50 Earth images while maintaining basic satellite health.

**Key Challenges**:
- Manage battery levels across 3 satellites
- Balance image capture with resource constraints
- Avoid satellite failures due to power depletion

### Success Criteria

| Criterion | Target | Description |
|-----------|--------|-------------|
| `min_images_captured` | 50 | Total images captured by all satellites |
| `min_avg_battery` | 30% | Average battery level across satellites |
| `max_satellite_failures` | 0 | Maximum allowed satellite failures |

### Strategy Guidelines

```python
def easy_task_strategy(observation):
    """Basic strategy for Easy task completion."""
    actions = {}

    for satellite in observation.satellites:
        # Simple resource-aware actions
        if satellite.battery < 20:
            # Critical: charge immediately
            actions[satellite.id] = "maintain"
        elif satellite.storage > 80:
            # High priority: clear storage
            actions[satellite.id] = "downlink"
        elif satellite.battery > 30 and satellite.storage < 70:
            # Good conditions: capture images
            actions[satellite.id] = "capture"
        else:
            # Default: wait
            actions[satellite.id] = "idle"

    return Action(satellite_actions=actions)
```

### Performance Benchmarks

- **Random Agent**: ~10-20 total reward
- **Basic Heuristic**: ~50-70 total reward
- **Optimal Policy**: ~80-100 total reward

## Medium Task

### Configuration

```python
class MediumTask(Task):
    name = "Medium"
    num_satellites = 5
    max_steps = 150
    success_criteria = {
        "min_images_captured": 120,
        "min_avg_battery": 0.5,  # 50%
        "min_data_transmitted": 80,  # % of captured data
        "max_satellite_failures": 1,
    }
```

### Objectives

**Primary Goal**: Capture 120+ images while maintaining 80% data transmission rate.

**Key Challenges**:
- Coordinate 5 satellites with interdependent resources
- Balance capture frequency with downlink opportunities
- Maintain higher average battery levels (50%)
- Allow maximum 1 satellite failure

### Success Criteria

| Criterion | Target | Description |
|-----------|--------|-------------|
| `min_images_captured` | 120 | Total images captured by all satellites |
| `min_avg_battery` | 50% | Average battery level across satellites |
| `min_data_transmitted` | 80% | Percentage of captured data successfully transmitted |
| `max_satellite_failures` | 1 | Maximum allowed satellite failures |

### Strategy Guidelines

```python
def medium_task_strategy(observation):
    """Coordinated strategy for Medium task."""
    actions = {}

    # Analyze constellation state
    avg_battery = sum(s.battery for s in observation.satellites) / len(observation.satellites)
    total_storage = sum(s.storage for s in observation.satellites)
    high_storage_count = sum(1 for s in observation.satellites if s.storage > 70)

    for satellite in observation.satellites:
        # Communication-aware decisions
        can_comm = any(distance_to_gs(satellite, gs) <= 500
                      for gs in observation.ground_stations)

        if satellite.battery < 25:
            actions[satellite.id] = "maintain"
        elif satellite.storage > 85 or (high_storage_count >= 3 and can_comm):
            actions[satellite.id] = "downlink"
        elif avg_battery > 40 and satellite.battery > 35 and satellite.storage < 60:
            actions[satellite.id] = "capture"
        else:
            actions[satellite.id] = "idle"

    return Action(satellite_actions=actions)
```

### Performance Benchmarks

- **Random Agent**: ~20-40 total reward
- **Basic Heuristic**: ~80-110 total reward
- **Optimal Policy**: ~130-160 total reward

## Hard Task

### Configuration

```python
class HardTask(Task):
    name = "Hard"
    num_satellites = 8
    max_steps = 200
    success_criteria = {
        "min_images_captured": 200,
        "min_avg_battery": 0.6,  # 60%
        "min_data_transmitted": 90,  # % of captured data
        "min_operational_rate": 0.75,  # 75% satellites operational
        "max_satellite_failures": 2,
    }
```

### Objectives

**Primary Goal**: Capture 200+ images with 90% data transmission and 75% satellite survival rate.

**Key Challenges**:
- Coordinate 8 satellites in complex orbital dynamics
- Maintain high operational tempo with strict resource constraints
- Achieve near-perfect data transmission rates
- Minimize satellite failures through proactive management
- Handle dynamic weather conditions affecting image quality

### Success Criteria

| Criterion | Target | Description |
|-----------|--------|-------------|
| `min_images_captured` | 200 | Total images captured by all satellites |
| `min_avg_battery` | 60% | Average battery level across satellites |
| `min_data_transmitted` | 90% | Percentage of captured data successfully transmitted |
| `min_operational_rate` | 75% | Minimum fraction of satellites remaining operational |
| `max_satellite_failures` | 2 | Maximum allowed satellite failures |

### Strategy Guidelines

```python
def hard_task_strategy(observation):
    """Advanced coordination strategy for Hard task."""
    actions = {}

    # Global constellation analysis
    satellites = observation.satellites
    operational = [s for s in satellites if s.operational]
    operational_rate = len(operational) / len(satellites)

    # Resource summary
    avg_battery = sum(s.battery for s in operational) / len(operational)
    total_storage_used = sum(s.storage for s in operational)
    high_storage_satellites = [s for s in operational if s.storage > 75]

    # Communication analysis
    comm_opportunities = {}
    for sat in operational:
        comm_opportunities[sat.id] = any(
            distance_to_gs(sat, gs) <= 500
            for gs in observation.ground_stations
        )

    # Role assignment
    imagers = []
    communicators = []
    chargers = []

    for sat in operational:
        if sat.storage > 85:
            communicators.append(sat)
        elif sat.battery < 30:
            chargers.append(sat)
        elif sat.battery > 50 and sat.storage < 50:
            imagers.append(sat)

    # Action assignment with coordination
    for sat in operational:
        if sat in chargers:
            actions[sat.id] = "maintain"
        elif sat in communicators and comm_opportunities[sat.id]:
            actions[sat.id] = "downlink"
        elif sat in imagers and get_weather_quality(observation) > 0.6:
            actions[sat.id] = "capture"
        elif len(high_storage_satellites) > 3 and comm_opportunities[sat.id]:
            # Help with communication if needed
            actions[sat.id] = "downlink"
        elif avg_battery < 45:
            # Contribute to charging if constellation needs it
            actions[sat.id] = "maintain"
        else:
            actions[sat.id] = "idle"

    return Action(satellite_actions=actions)
```

### Performance Benchmarks

- **Random Agent**: ~30-60 total reward
- **Basic Heuristic**: ~120-160 total reward
- **Optimal Policy**: ~200-250 total reward

## Task Evaluation

### Grading System

Tasks are evaluated using partial credit grading:

```python
def grade_task_performance(episode_history, task):
    """Grade task performance with partial credit."""
    final_state = episode_history[-1]
    success_criteria = task.success_criteria

    scores = {}

    # Images captured
    total_images = sum(1 for step in episode_history
                      for result in step.action_results
                      if result.action == "capture" and result.success)
    target_images = success_criteria["min_images_captured"]
    scores["images_captured"] = min(1.0, total_images / target_images)

    # Average battery
    avg_battery = sum(s.battery for s in final_state.satellites) / len(final_state.satellites)
    target_battery = success_criteria["min_avg_battery"]
    scores["avg_battery"] = min(1.0, avg_battery / target_battery)

    # Data transmission rate
    if "min_data_transmitted" in success_criteria:
        total_captured = sum(result.data_amount for step in episode_history
                           for result in step.action_results
                           if result.action == "capture" and result.success)
        total_transmitted = sum(result.data_amount for step in episode_history
                              for result in step.action_results
                              if result.action == "downlink" and result.success)
        transmission_rate = total_transmitted / max(1, total_captured)
        target_rate = success_criteria["min_data_transmitted"] / 100.0
        scores["data_transmitted"] = min(1.0, transmission_rate / target_rate)

    # Operational rate
    if "min_operational_rate" in success_criteria:
        operational_count = sum(1 for s in final_state.satellites if s.operational)
        operational_rate = operational_count / len(final_state.satellites)
        target_rate = success_criteria["min_operational_rate"]
        scores["operational_rate"] = min(1.0, operational_rate / target_rate)

    # Satellite failures
    if "max_satellite_failures" in success_criteria:
        failed_count = sum(1 for s in final_state.satellites if not s.operational)
        max_failures = success_criteria["max_satellite_failures"]
        if failed_count <= max_failures:
            scores["satellite_failures"] = 1.0
        else:
            scores["satellite_failures"] = max(0.0, 1.0 - (failed_count - max_failures) / max_failures)

    # Overall score
    overall_score = sum(scores.values()) / len(scores)

    return overall_score, scores
```

### Success Determination

```python
def is_task_successful(episode_score, task):
    """Determine if task success criteria are met."""
    # Require minimum overall score
    min_overall_score = 0.8  # 80% of criteria met

    # All critical criteria must be met
    critical_criteria = ["min_images_captured", "max_satellite_failures"]
    for criterion in critical_criteria:
        if episode_score.get(criterion, 0.0) < 0.8:
            return False

    return episode_score.overall >= min_overall_score
```

## Task Progression

### Difficulty Scaling

```python
def get_task_progression():
    """Define task progression for curriculum learning."""
    return {
        "Easy": {
            "num_satellites": 3,
            "max_steps": 100,
            "complexity": 1.0,
        },
        "Medium": {
            "num_satellites": 5,
            "max_steps": 150,
            "complexity": 2.5,
        },
        "Hard": {
            "num_satellites": 8,
            "max_steps": 200,
            "complexity": 4.0,
        }
    }
```

### Transfer Learning

```python
def adapt_strategy_for_task(base_strategy, target_task):
    """Adapt strategy parameters for different task difficulties."""
    task_multipliers = {
        "Easy": {"action_frequency": 0.7, "risk_tolerance": 0.8},
        "Medium": {"action_frequency": 0.85, "risk_tolerance": 0.6},
        "Hard": {"action_frequency": 1.0, "risk_tolerance": 0.4},
    }

    multipliers = task_multipliers[target_task.name]

    def adapted_strategy(observation):
        # Apply task-specific adjustments
        actions = base_strategy(observation)

        # Scale action frequency
        for sat_id in actions.satellite_actions:
            if random.random() > multipliers["action_frequency"]:
                actions.satellite_actions[sat_id] = "idle"

        return actions

    return adapted_strategy
```

## Task Customization

### Creating Custom Tasks

```python
def create_custom_task(name, config):
    """Create a custom task with specific parameters."""

    class CustomTask(Task):
        def __init__(self):
            self.name = name
            self.num_satellites = config.get("num_satellites", 4)
            self.max_steps = config.get("max_steps", 120)
            self.success_criteria = config.get("success_criteria", {
                "min_images_captured": 80,
                "min_avg_battery": 0.4,
            })

        def setup_environment(self, env):
            # Custom environment setup
            env.num_satellites = self.num_satellites
            env.max_steps = self.max_steps
            # Additional customizations...

        def get_success_criteria(self):
            return self.success_criteria

    return CustomTask()
```

### Task Validation

```python
def validate_task_config(task):
    """Validate task configuration for consistency."""
    errors = []

    # Basic validation
    if task.num_satellites < 1:
        errors.append("Must have at least 1 satellite")

    if task.max_steps < 10:
        errors.append("Episode must be at least 10 steps")

    # Success criteria validation
    required_criteria = ["min_images_captured"]
    for criterion in required_criteria:
        if criterion not in task.success_criteria:
            errors.append(f"Missing required criterion: {criterion}")

    # Logical consistency
    if "min_data_transmitted" in task.success_criteria:
        if "min_images_captured" not in task.success_criteria:
            errors.append("Data transmission requires image capture criterion")

    return errors
```

---

[← Reward Function](reward-function.md) | [API Reference →](../api-reference/index.md)