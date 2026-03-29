# Reward Function

Detailed explanation of the reward system, components, and optimization strategies.

## Reward Structure

Rewards are returned as `Reward` Pydantic models with total value and component breakdown:

```python
from satellite_env import Reward

# Example reward structure
reward = Reward(
    value=15.5,  # Total reward value
    components={
        "capture_success": 10.0,
        "downlink_success": 2.0,
        "maintenance_bonus": 5.0,
        "time_penalty": -1.5,
    }
)
```

## Reward Components

### 1. Action Success Rewards

#### Capture Success (`capture_success`)

**Description**: Reward for successfully capturing Earth images.

**Calculation**:
```python
capture_reward = 10.0 * weather_quality_factor
```

**Factors**:
- **Base Reward**: 10.0 points for successful capture
- **Weather Quality**: Multiplier based on cloud cover (0.1 to 1.0)
- **Resource Validation**: Must have sufficient battery (≥5%) and storage (≤90%)

**Examples**:
```python
# Perfect weather conditions
weather_quality = 1.0
capture_reward = 10.0 * 1.0  # = 10.0

# Moderate cloud cover (50%)
weather_quality = 0.5
capture_reward = 10.0 * 0.5  # = 5.0

# Heavy cloud cover (90%)
weather_quality = 0.1
capture_reward = 10.0 * 0.1  # = 1.0
```

#### Downlink Success (`downlink_success`)

**Description**: Reward for successfully transmitting data to ground stations.

**Calculation**:
```python
data_transmitted = min(20.0, current_storage)  # Max 20% per downlink
downlink_reward = 2.0 * data_transmitted
```

**Factors**:
- **Base Rate**: 2.0 points per percentage of storage cleared
- **Data Amount**: Proportional to storage transmitted (0-20%)
- **Communication Range**: Must be within 500km of ground station
- **Resource Validation**: Must have sufficient battery (≥2%) and data to transmit

**Examples**:
```python
# Transmit 15% of storage
data_transmitted = 15.0
downlink_reward = 2.0 * 15.0  # = 30.0

# Transmit 5% of storage
data_transmitted = 5.0
downlink_reward = 2.0 * 5.0   # = 10.0
```

#### Maintenance Bonus (`maintenance_bonus`)

**Description**: Reward for performing maintenance operations (primarily battery charging).

**Calculation**:
```python
maintenance_reward = 5.0
```

**Factors**:
- **Fixed Reward**: 5.0 points for any maintenance action
- **Always Available**: No resource requirements
- **Strategic Value**: Encourages proactive resource management

### 2. Penalty Components

#### Action Failure Penalty (`action_failure`)

**Description**: Penalty for attempting invalid actions.

**Calculation**:
```python
failure_penalty = -1.0
```

**Trigger Conditions**:
- Insufficient battery for capture (< 5%)
- Insufficient battery for downlink (< 2%)
- Insufficient storage for downlink (0%)
- Attempting to exceed storage capacity (> 90% for capture)
- Communication out of range for downlink (> 500km)

**Examples**:
```python
# Attempt capture with 3% battery (need 5%)
action = "capture"
battery = 3.0
is_valid = battery >= 5.0  # False
failure_penalty = -1.0

# Attempt downlink with no data
action = "downlink"
storage = 0.0
is_valid = storage > 0.0  # False
failure_penalty = -1.0
```

#### Time Penalty (`time_penalty`)

**Description**: Penalty applied each time step to encourage efficiency.

**Calculation**:
```python
time_penalty = -0.1 * step_count
```

**Factors**:
- **Linear Decay**: Increases in magnitude over time
- **Efficiency Incentive**: Rewards faster task completion
- **Task Scaling**: Same rate across Easy/Medium/Hard tasks

**Examples**:
```python
# Early in episode (step 10)
step_count = 10
time_penalty = -0.1 * 10  # = -1.0

# Later in episode (step 50)
step_count = 50
time_penalty = -0.1 * 50  # = -5.0
```

### 3. Task-Specific Rewards

#### Success Bonus (`success_bonus`)

**Description**: Bonus reward for achieving task success criteria.

**Calculation**:
```python
success_bonus = 100.0 * task_completion_rate
```

**Factors**:
- **Base Bonus**: 100.0 points maximum
- **Completion Rate**: Percentage of success criteria met (0.0 to 1.0)
- **Task Dependent**: Only awarded at episode end if criteria satisfied

**Examples**:
```python
# Easy task: Capture 50 images, achieved 45
completion_rate = 45/50  # = 0.9
success_bonus = 100.0 * 0.9  # = 90.0

# Medium task: 80% average battery, achieved 85%
completion_rate = 0.85
success_bonus = 100.0 * 0.85  # = 85.0
```

#### Failure Penalty (`failure_penalty`)

**Description**: Penalty for failing to meet minimum task requirements.

**Calculation**:
```python
failure_penalty = -50.0
```

**Trigger Conditions**:
- Episode ends without meeting minimum success criteria
- Satellite failures or resource depletion
- Task timeout without completion

## Total Reward Calculation

### Step Reward Formula

```python
def calculate_step_reward(action_results, step_count):
    """Calculate reward for a single time step."""
    total_reward = 0.0
    components = {}

    # Action success rewards
    for result in action_results:
        if result.success:
            if result.action == "capture":
                weather_factor = get_weather_quality_factor()
                reward = 10.0 * weather_factor
                components["capture_success"] = components.get("capture_success", 0) + reward
            elif result.action == "downlink":
                data_amount = result.data_transmitted
                reward = 2.0 * data_amount
                components["downlink_success"] = components.get("downlink_success", 0) + reward
            elif result.action == "maintain":
                components["maintenance_bonus"] = components.get("maintenance_bonus", 0) + 5.0
        else:
            components["action_failure"] = components.get("action_failure", 0) - 1.0

    # Time penalty
    time_penalty = -0.1 * step_count
    components["time_penalty"] = time_penalty

    # Sum components
    total_reward = sum(components.values())

    return Reward(value=total_reward, components=components)
```

### Episode Reward Formula

```python
def calculate_episode_reward(step_rewards, task_success):
    """Calculate final episode reward including task completion."""
    total_reward = sum(reward.value for reward in step_rewards)
    components = {}

    # Aggregate step components
    for reward in step_rewards:
        for component, value in reward.components.items():
            components[component] = components.get(component, 0) + value

    # Task completion bonus/penalty
    if task_success:
        completion_rate = calculate_task_completion_rate()
        success_bonus = 100.0 * completion_rate
        components["success_bonus"] = success_bonus
        total_reward += success_bonus
    else:
        components["failure_penalty"] = -50.0
        total_reward -= 50.0

    return Reward(value=total_reward, components=components)
```

## Reward Optimization Strategies

### Maximize Capture Efficiency

```python
def optimize_capture_strategy(observation):
    """Strategy to maximize capture rewards."""
    actions = {}

    for satellite in observation.satellites:
        # Prioritize good weather and available resources
        weather_good = get_avg_weather_quality(observation) > 0.7
        can_capture = (satellite.battery >= 5 and satellite.storage <= 90)

        if weather_good and can_capture:
            actions[satellite.id] = "capture"
        elif satellite.battery < 20:
            actions[satellite.id] = "maintain"
        else:
            actions[satellite.id] = "idle"

    return Action(satellite_actions=actions)
```

### Balance Resource Management

```python
def balanced_resource_strategy(observation):
    """Strategy balancing capture and resource management."""
    actions = {}

    for satellite in observation.satellites:
        # Assess priorities
        needs_charge = satellite.battery < 30
        needs_downlink = satellite.storage > 80
        can_capture = satellite.battery >= 5 and satellite.storage <= 90
        weather_ok = get_avg_weather_quality(observation) > 0.5

        if needs_charge:
            actions[satellite.id] = "maintain"
        elif needs_downlink:
            actions[satellite.id] = "downlink"
        elif can_capture and weather_ok:
            actions[satellite.id] = "capture"
        else:
            actions[satellite.id] = "idle"

    return Action(satellite_actions=actions)
```

### Time-Efficient Completion

```python
def time_efficient_strategy(observation):
    """Strategy prioritizing task completion speed."""
    actions = {}
    steps_remaining = observation.task_info.max_steps - observation.step_count
    urgency = 1.0 - (steps_remaining / observation.task_info.max_steps)

    for satellite in observation.satellites:
        # Increase action frequency as deadline approaches
        action_probability = 0.3 + (urgency * 0.7)  # 30% to 100%

        if random.random() < action_probability:
            if satellite.battery >= 5 and satellite.storage <= 90:
                actions[satellite.id] = "capture"
            elif satellite.storage > 70:
                actions[satellite.id] = "downlink"
            elif satellite.battery < 50:
                actions[satellite.id] = "maintain"
            else:
                actions[satellite.id] = "idle"
        else:
            actions[satellite.id] = "idle"

    return Action(satellite_actions=actions)
```

## Reward Analysis and Debugging

### Component Breakdown Analysis

```python
def analyze_reward_components(rewards):
    """Analyze reward composition across episode."""
    component_totals = {}
    component_counts = {}

    for reward in rewards:
        for component, value in reward.components.items():
            component_totals[component] = component_totals.get(component, 0) + value
            component_counts[component] = component_counts.get(component, 0) + 1

    # Calculate averages
    analysis = {}
    for component in component_totals:
        analysis[component] = {
            'total': component_totals[component],
            'average': component_totals[component] / component_counts[component],
            'count': component_counts[component],
        }

    return analysis
```

### Reward Trend Visualization

```python
def plot_reward_trends(rewards):
    """Visualize reward trends over episode."""
    steps = range(len(rewards))
    total_rewards = [r.value for r in rewards]

    # Component tracking
    components_over_time = {}
    for reward in rewards:
        for component, value in reward.components.items():
            if component not in components_over_time:
                components_over_time[component] = []
            components_over_time[component].append(value)

    # Plotting code would go here
    # plt.plot(steps, total_rewards, label='Total Reward')
    # for component, values in components_over_time.items():
    #     plt.plot(steps, values, label=component, alpha=0.7)

    return components_over_time
```

### Reward Effectiveness Metrics

```python
def calculate_reward_effectiveness(episode_rewards, task_success):
    """Calculate how effectively rewards guided behavior."""
    metrics = {}

    # Reward density (reward per step)
    total_reward = sum(r.value for r in episode_rewards)
    metrics['reward_density'] = total_reward / len(episode_rewards)

    # Success efficiency (reward per success criteria met)
    if task_success:
        completion_rate = calculate_task_completion_rate()
        metrics['success_efficiency'] = total_reward / completion_rate
    else:
        metrics['success_efficiency'] = 0.0

    # Action success rate
    action_attempts = sum(len(r.components) for r in episode_rewards
                         if 'action_failure' in r.components)
    action_failures = sum(r.components.get('action_failure', 0) for r in episode_rewards)
    metrics['action_success_rate'] = 1.0 - (action_failures / action_attempts) if action_attempts > 0 else 1.0

    # Resource efficiency
    capture_rewards = sum(r.components.get('capture_success', 0) for r in episode_rewards)
    downlink_rewards = sum(r.components.get('downlink_success', 0) for r in episode_rewards)
    maintenance_rewards = sum(r.components.get('maintenance_bonus', 0) for r in episode_rewards)

    metrics['resource_efficiency'] = (capture_rewards + downlink_rewards) / max(1, maintenance_rewards)

    return metrics
```

## Advanced Reward Features

### Shaped Rewards

```python
def add_shaped_rewards(base_reward, observation, action):
    """Add shaped rewards to guide learning."""
    shaped_components = {}

    # Distance to ground station bonus
    for sat_id, act in action.satellite_actions.items():
        if act == "downlink":
            satellite = observation.satellites[sat_id]
            nearest_dist = min(distance_to_gs(satellite, gs)
                             for gs in observation.ground_stations)
            if nearest_dist <= 500:
                # Bonus for being in range
                shaped_components[f"comm_range_bonus_{sat_id}"] = 1.0

    # Resource management bonuses
    battery_avg = sum(s.battery for s in observation.satellites) / len(observation.satellites)
    if battery_avg > 70:
        shaped_components["high_battery_bonus"] = 2.0

    storage_avg = sum(s.storage for s in observation.satellites) / len(observation.satellites)
    if storage_avg < 50:
        shaped_components["low_storage_bonus"] = 1.0

    # Add shaped rewards to base
    shaped_total = sum(shaped_components.values())
    base_reward.value += shaped_total
    base_reward.components.update(shaped_components)

    return base_reward
```

### Multi-Objective Rewards

```python
def multi_objective_rewards(observation, action_results):
    """Calculate rewards for multiple objectives."""
    objectives = {
        'data_collection': 0.0,
        'resource_management': 0.0,
        'communication_efficiency': 0.0,
        'task_completion': 0.0,
    }

    # Data collection objective
    capture_successes = sum(1 for r in action_results
                           if r.action == "capture" and r.success)
    objectives['data_collection'] = capture_successes * 10.0

    # Resource management objective
    battery_levels = [s.battery for s in observation.satellites]
    storage_levels = [s.storage for s in observation.satellites]
    battery_avg = sum(battery_levels) / len(battery_levels)
    storage_avg = sum(storage_levels) / len(storage_levels)

    objectives['resource_management'] = (battery_avg + (100 - storage_avg)) / 2

    # Communication efficiency objective
    downlink_successes = sum(1 for r in action_results
                            if r.action == "downlink" and r.success)
    objectives['communication_efficiency'] = downlink_successes * 5.0

    # Task completion objective (calculated at episode end)
    # objectives['task_completion'] = calculate_task_completion_score()

    return objectives
```

### Adaptive Reward Scaling

```python
class AdaptiveRewardScaler:
    """Adapt reward scaling based on agent performance."""

    def __init__(self, target_reward=50.0, adaptation_rate=0.1):
        self.target_reward = target_reward
        self.adaptation_rate = adaptation_rate
        self.scale_factor = 1.0

    def adapt_scale(self, episode_reward):
        """Adapt scale factor based on recent performance."""
        reward_diff = episode_reward - self.target_reward
        scale_adjustment = self.adaptation_rate * (reward_diff / self.target_reward)
        self.scale_factor = max(0.1, min(5.0, self.scale_factor + scale_adjustment))

    def scale_reward(self, reward):
        """Apply adaptive scaling to reward."""
        scaled_value = reward.value * self.scale_factor
        scaled_components = {k: v * self.scale_factor
                           for k, v in reward.components.items()}

        return Reward(value=scaled_value, components=scaled_components)
```

---

[← Observation Space](observation-space.md) | [Tasks →](tasks.md)