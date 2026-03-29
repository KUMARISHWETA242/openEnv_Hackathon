# Action Space

Complete description of available actions, their effects, and usage patterns.

## Action Format

Actions are specified using the `Action` Pydantic model:

```python
from satellite_env import Action

action = Action(satellite_actions={
    satellite_id: action_name,
    ...
})
```

Where:
- `satellite_id`: Integer identifier of the satellite (0 to num_satellites-1)
- `action_name`: String specifying the action to perform

## Available Actions

### 1. Capture (`"capture"`)

**Description**: Take an Earth image using the satellite's camera system.

**Requirements**:
- Battery level ≥ 5%
- Storage utilization ≤ 90%
- Satellite operational

**Effects**:
- Battery: `-5%`
- Storage: `+10%`
- Reward: `+10` (success) or `-1` (failure)

**Use Cases**:
- Primary mission objective for imaging satellites
- Best performed in good weather conditions
- Balances resource consumption with data collection

```python
# Example: Command satellite 0 to capture an image
action = Action(satellite_actions={0: "capture"})
```

### 2. Downlink (`"downlink"`)

**Description**: Transmit stored data to a ground station.

**Requirements**:
- Battery level ≥ 2%
- Storage utilization > 0%
- Satellite within range of a ground station (≤ 500 km)

**Effects**:
- Battery: `-2%`
- Storage: `-min(20%, current_storage)`
- Reward: `+2 × data_sent` (success) or `-1` (failure)

**Use Cases**:
- Clear storage when full
- Transmit valuable data to ground
- Required for data-dependent tasks

```python
# Example: Command satellite 1 to downlink data
action = Action(satellite_actions={1: "downlink"})
```

### 3. Maintain (`"maintain"`)

**Description**: Perform maintenance operations, primarily battery charging.

**Requirements**:
- None (always available)

**Effects**:
- Battery: `+20%` (capped at 100%)
- Storage: No change
- Reward: `+5`

**Use Cases**:
- Recharge battery when low
- Prepare for high-energy operations
- Prevent satellite failure due to power loss

```python
# Example: Command satellite 2 to perform maintenance
action = Action(satellite_actions={2: "maintain"})
```

### 4. Idle (`"idle"`)

**Description**: No operation - satellite remains in current state.

**Requirements**:
- None (always available)

**Effects**:
- Battery: No change
- Storage: No change
- Reward: `0`

**Use Cases**:
- Conserve resources when no action is beneficial
- Default action when other actions are invalid
- Strategic waiting for better conditions

```python
# Example: Command satellite 3 to idle
action = Action(satellite_actions={3: "idle"})
```

## Action Validation

### Automatic Validation

The environment automatically validates actions before execution:

```python
def validate_action(satellite: SatelliteState, action: str) -> bool:
    """Validate if an action can be performed by a satellite."""
    if action == "capture":
        return satellite.battery >= 5 and satellite.storage <= 90
    elif action == "downlink":
        return (satellite.battery >= 2 and
                satellite.storage > 0 and
                can_downlink(satellite.position))
    elif action == "maintain":
        return True
    elif action == "idle":
        return True
    return False
```

### Invalid Action Handling

When an invalid action is attempted:
- Action is not executed
- Satellite remains in current state
- Penalty reward of `-1` is applied
- Episode continues normally

```python
# Example of invalid action (low battery for capture)
satellite.battery = 3  # Below 5% requirement
action = Action(satellite_actions={0: "capture"})

# Result: action fails, reward = -1
observation, reward, done, info = env.step(action)
assert reward.value == -1
```

## Action Selection Strategies

### Resource-Aware Selection

```python
def resource_aware_actions(observation):
    """Select actions based on current resource levels."""
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

### Communication-Aware Selection

```python
def communication_aware_actions(observation):
    """Prioritize downlinking when in communication range."""
    actions = {}
    for satellite in observation.satellites:
        # Check communication availability
        can_comm = any(
            ((satellite.position[0] - gs[0])**2 +
             (satellite.position[1] - gs[1])**2) < 500**2
            for gs in observation.ground_stations
        )

        if satellite.storage > 80 and can_comm:
            actions[satellite.id] = "downlink"  # High priority
        elif satellite.battery > 25:
            actions[satellite.id] = "capture"  # Normal operation
        else:
            actions[satellite.id] = "maintain"  # Recharge
    return Action(satellite_actions=actions)
```

### Weather-Aware Selection

```python
def weather_aware_actions(observation):
    """Consider weather conditions for imaging decisions."""
    avg_cloud_cover = sum(observation.weather_conditions.values()) / len(observation.weather_conditions)

    actions = {}
    for satellite in observation.satellites:
        if avg_cloud_cover < 0.3 and satellite.battery > 20:
            actions[satellite.id] = "capture"  # Good conditions
        elif satellite.battery < 50:
            actions[satellite.id] = "maintain"  # Recharge
        elif satellite.storage > 70:
            actions[satellite.id] = "downlink"  # Clear storage
        else:
            actions[satellite.id] = "idle"     # Wait
    return Action(satellite_actions=actions)
```

## Action Timing Considerations

### Sequential Dependencies

Some action sequences are more effective than others:

```python
# Good sequence: Capture -> Downlink -> Maintain
actions_sequence = [
    Action(satellite_actions={0: "capture"}),   # Fill storage
    Action(satellite_actions={0: "downlink"}),  # Clear storage
    Action(satellite_actions={0: "maintain"}),  # Recharge battery
]

# Less efficient: Repeated capture without downlinking
inefficient_sequence = [
    Action(satellite_actions={0: "capture"}),   # Storage: 10%
    Action(satellite_actions={0: "capture"}),   # Storage: 20%
    Action(satellite_actions={0: "capture"}),   # Storage: 30%
    # ... eventually can't capture anymore
]
```

### Time Step Effects

Actions have timing implications:

- **Battery Drain**: 0.5% per time step regardless of action
- **Position Changes**: Satellites move between time steps
- **Communication Windows**: Availability changes with position
- **Weather Changes**: Conditions may improve/degrade over time

## Multi-Satellite Coordination

### Independent Actions

Satellites can act independently:

```python
# Each satellite performs different actions
action = Action(satellite_actions={
    0: "capture",   # Imaging satellite
    1: "downlink",  # Communication satellite
    2: "maintain",  # Charging satellite
    3: "idle",      # Standby satellite
})
```

### Coordinated Strategies

For complex tasks, coordinate satellite actions:

```python
def coordinated_strategy(observation):
    """Coordinate satellites for team objectives."""
    actions = {}

    # Designate roles based on satellite state
    imagers = []
    communicators = []
    chargers = []

    for satellite in observation.satellites:
        if satellite.storage < 30:
            imagers.append(satellite.id)
        elif satellite.storage > 70:
            communicators.append(satellite.id)
        elif satellite.battery < 40:
            chargers.append(satellite.id)

    # Assign actions based on roles
    for sat_id in imagers[:2]:  # Limit imagers
        actions[sat_id] = "capture"
    for sat_id in communicators:
        actions[sat_id] = "downlink"
    for sat_id in chargers:
        actions[sat_id] = "maintain"

    # Fill remaining with idle
    for satellite in observation.satellites:
        if satellite.id not in actions:
            actions[satellite.id] = "idle"

    return Action(satellite_actions=actions)
```

## Performance Optimization

### Action Batching

For efficiency with many satellites:

```python
def batch_actions(observation, strategy_function):
    """Apply the same strategy logic to all satellites."""
    actions = {}
    for satellite in observation.satellites:
        # Apply strategy to each satellite independently
        action = strategy_function(satellite, observation)
        actions[satellite.id] = action
    return Action(satellite_actions=actions)
```

### Caching Valid Actions

Pre-compute valid actions to avoid repeated validation:

```python
def get_valid_actions(satellite, observation):
    """Return list of valid actions for a satellite."""
    valid = []
    for action in ["capture", "downlink", "maintain", "idle"]:
        if validate_action(satellite, action, observation):
            valid.append(action)
    return valid
```

## Debugging Actions

### Action Logging

Track action execution for debugging:

```python
def log_action_execution(action, observation, reward):
    """Log detailed action information."""
    print(f"Executed actions:")
    for sat_id, act in action.satellite_actions.items():
        satellite = observation.satellites[sat_id]
        print(f"  Satellite {sat_id} ({act}): "
              f"Battery: {satellite.battery:.1f}%, "
              f"Storage: {satellite.storage:.1f}%")

    print(f"Total reward: {reward.value}")
    if hasattr(reward, 'components'):
        for component, value in reward.components.items():
            print(f"  {component}: {value}")
```

### Action Validation Testing

Test action validation logic:

```python
def test_action_validation():
    """Test action validation with various states."""
    test_cases = [
        {"battery": 10, "storage": 50, "action": "capture", "expected": True},
        {"battery": 3, "storage": 50, "action": "capture", "expected": False},
        {"battery": 10, "storage": 95, "action": "capture", "expected": False},
        {"battery": 5, "storage": 50, "action": "downlink", "expected": True},
        {"battery": 1, "storage": 50, "action": "downlink", "expected": False},
    ]

    for case in test_cases:
        # Create mock satellite
        satellite = type('MockSatellite', (), case)()
        result = validate_action(satellite, case["action"])
        assert result == case["expected"], f"Failed for {case}"
```

---

[← Environment Overview](overview.md) | [Observation Space →](observation-space.md)