# Observation Space

Complete description of the observation structure, data types, and interpretation.

## Observation Format

Observations are returned as `Observation` Pydantic models:

```python
from satellite_env import Observation

# Example observation structure
observation = Observation(
    satellites=[SatelliteState(...)],
    ground_stations=[GroundStation(...)],
    weather_conditions={region: cloud_cover},
    step_count=42,
    task_info=TaskInfo(...)
)
```

## Satellite State

### Core Properties

Each satellite provides the following state information:

```python
class SatelliteState(BaseModel):
    id: int                    # Unique satellite identifier (0 to num_satellites-1)
    position: Tuple[float, float]  # (x, y) coordinates in km
    battery: float             # Battery level (0.0 to 100.0)
    storage: float             # Storage utilization (0.0 to 100.0)
    operational: bool          # Whether satellite is functional
    last_action: Optional[str] # Last action performed ("capture", "downlink", "maintain", "idle")
```

### Position Information

```python
# Position coordinates
satellite.position[0]  # X coordinate (km from origin)
satellite.position[1]  # Y coordinate (km from origin)

# Distance calculations
def distance_to_ground_station(satellite, ground_station):
    """Calculate distance between satellite and ground station."""
    dx = satellite.position[0] - ground_station.position[0]
    dy = satellite.position[1] - ground_station.position[1]
    return (dx**2 + dy**2)**0.5

# Communication range check
COMMUNICATION_RANGE = 500  # km
can_communicate = distance_to_ground_station(satellite, gs) <= COMMUNICATION_RANGE
```

### Resource Levels

```python
# Battery management
battery_critical = satellite.battery < 20.0    # Need immediate charging
battery_low = satellite.battery < 50.0         # Consider charging soon
battery_good = satellite.battery >= 75.0       # Ready for energy-intensive actions

# Storage management
storage_full = satellite.storage > 90.0        # Need to downlink soon
storage_high = satellite.storage > 70.0        # Consider downlinking
storage_low = satellite.storage < 30.0         # Good for capturing
```

### Operational Status

```python
# Satellite health checks
satellite.operational  # True if satellite is working

# Action history
if satellite.last_action == "capture":
    # Satellite just took an image
elif satellite.last_action == "downlink":
    # Satellite just transmitted data
elif satellite.last_action == "maintain":
    # Satellite just performed maintenance
elif satellite.last_action == "idle":
    # Satellite did nothing last step
```

## Ground Stations

### Station Properties

Ground stations provide communication infrastructure:

```python
class GroundStation(BaseModel):
    id: int                    # Unique station identifier
    position: Tuple[float, float]  # (x, y) coordinates in km
    name: str                  # Human-readable name
```

### Communication Capabilities

```python
# Ground station locations
ground_stations = observation.ground_stations

# Find nearest ground station
def find_nearest_ground_station(satellite, ground_stations):
    """Return the closest ground station to a satellite."""
    min_distance = float('inf')
    nearest = None

    for gs in ground_stations:
        distance = distance_to_ground_station(satellite, gs)
        if distance < min_distance:
            min_distance = distance
            nearest = gs

    return nearest, min_distance

# Check communication availability
nearest_gs, distance = find_nearest_ground_station(satellite, ground_stations)
can_downlink = distance <= COMMUNICATION_RANGE
```

## Weather Conditions

### Weather Data Structure

Weather conditions affect imaging quality:

```python
# Weather conditions dictionary
weather_conditions = observation.weather_conditions  # Dict[str, float]

# Keys: region names (e.g., "region_0", "region_1", ...)
# Values: cloud cover percentage (0.0 to 1.0)
# 0.0 = clear skies (perfect imaging)
# 1.0 = complete cloud cover (no imaging possible)
```

### Weather Impact on Actions

```python
# Weather-aware decision making
def get_imaging_quality(weather_conditions):
    """Calculate overall imaging quality based on weather."""
    if not weather_conditions:
        return 1.0  # Default to perfect conditions

    # Average cloud cover across all regions
    avg_cloud_cover = sum(weather_conditions.values()) / len(weather_conditions)
    return 1.0 - avg_cloud_cover  # Quality decreases with cloud cover

# Action success probability
imaging_quality = get_imaging_quality(observation.weather_conditions)
capture_success_probability = max(0.1, imaging_quality)  # Minimum 10% success rate
```

### Weather Prediction

```python
# Weather evolution (simplified model)
def predict_weather_evolution(current_weather, steps_ahead=1):
    """Predict weather changes over time steps."""
    # Weather changes slowly (±5% per step)
    predicted = {}
    for region, cloud_cover in current_weather.items():
        # Random walk with bounds
        change = random.uniform(-0.05, 0.05)
        predicted[region] = max(0.0, min(1.0, cloud_cover + change))
    return predicted
```

## Step Information

### Step Counter

```python
# Episode progress tracking
current_step = observation.step_count
max_steps = observation.task_info.max_steps
steps_remaining = max_steps - current_step

# Episode completion
episode_progress = current_step / max_steps
time_pressure = 1.0 - (steps_remaining / max_steps)  # Increases over time
```

### Task Context

```python
# Task information
task_info = observation.task_info

# Available task properties
task_name = task_info.name          # "Easy", "Medium", or "Hard"
max_steps = task_info.max_steps     # Maximum episode length
num_satellites = task_info.num_satellites  # Number of satellites
success_criteria = task_info.success_criteria  # Task-specific goals
```

## Complete Observation Example

```python
# Full observation from environment
observation = env.reset()

print("=== SATELLITE CONSTELLATION OBSERVATION ===")
print(f"Step: {observation.step_count}")
print(f"Task: {observation.task_info.name}")
print(f"Max Steps: {observation.task_info.max_steps}")
print()

print("SATELLITES:")
for satellite in observation.satellites:
    print(f"  ID {satellite.id}:")
    print(f"    Position: ({satellite.position[0]:.1f}, {satellite.position[1]:.1f}) km")
    print(f"    Battery: {satellite.battery:.1f}%")
    print(f"    Storage: {satellite.storage:.1f}%")
    print(f"    Operational: {satellite.operational}")
    print(f"    Last Action: {satellite.last_action or 'None'}")
print()

print("GROUND STATIONS:")
for gs in observation.ground_stations:
    print(f"  {gs.name} (ID {gs.id}): ({gs.position[0]:.1f}, {gs.position[1]:.1f}) km")
print()

print("WEATHER CONDITIONS:")
for region, cloud_cover in observation.weather_conditions.items():
    print(f"  {region}: {cloud_cover:.1f} ({cloud_cover*100:.0f}% cloud cover)")
print()

print("TASK SUCCESS CRITERIA:")
for criterion, target in observation.task_info.success_criteria.items():
    print(f"  {criterion}: {target}")
```

## Observation Processing

### State Vector Extraction

```python
def observation_to_vector(observation):
    """Convert observation to flat vector for ML models."""
    vectors = []

    # Satellite features (4 features per satellite)
    for satellite in observation.satellites:
        sat_vector = [
            satellite.position[0] / 1000,  # Normalize position
            satellite.position[1] / 1000,
            satellite.battery / 100,       # Normalize to 0-1
            satellite.storage / 100,
        ]
        vectors.extend(sat_vector)

    # Ground station features (2 features per station)
    for gs in observation.ground_stations:
        gs_vector = [
            gs.position[0] / 1000,
            gs.position[1] / 1000,
        ]
        vectors.extend(gs_vector)

    # Weather features
    weather_vector = list(observation.weather_conditions.values())
    vectors.extend(weather_vector)

    # Step information
    step_vector = [
        observation.step_count / observation.task_info.max_steps,  # Progress
    ]
    vectors.extend(step_vector)

    return vectors
```

### Distance Matrix Calculation

```python
def calculate_distance_matrix(observation):
    """Calculate distances between all satellites and ground stations."""
    satellites = observation.satellites
    ground_stations = observation.ground_stations

    # Satellite-to-satellite distances
    sat_distances = {}
    for i, sat1 in enumerate(satellites):
        for j, sat2 in enumerate(satellites):
            if i != j:
                dist = ((sat1.position[0] - sat2.position[0])**2 +
                       (sat1.position[1] - sat2.position[1])**2)**0.5
                sat_distances[(sat1.id, sat2.id)] = dist

    # Satellite-to-ground-station distances
    comm_distances = {}
    for sat in satellites:
        for gs in ground_stations:
            dist = ((sat.position[0] - gs.position[0])**2 +
                   (sat.position[1] - gs.position[1])**2)**0.5
            comm_distances[(sat.id, gs.id)] = dist

    return sat_distances, comm_distances
```

### Resource Summary

```python
def get_resource_summary(observation):
    """Summarize resource levels across constellation."""
    satellites = observation.satellites

    # Battery statistics
    batteries = [s.battery for s in satellites]
    battery_stats = {
        'mean': sum(batteries) / len(batteries),
        'min': min(batteries),
        'max': max(batteries),
        'critical_count': sum(1 for b in batteries if b < 20),
        'low_count': sum(1 for b in batteries if b < 50),
    }

    # Storage statistics
    storages = [s.storage for s in satellites]
    storage_stats = {
        'mean': sum(storages) / len(storages),
        'min': min(storages),
        'max': max(storages),
        'full_count': sum(1 for s in storages if s > 90),
        'high_count': sum(1 for s in storages if s > 70),
    }

    # Operational status
    operational_count = sum(1 for s in satellites if s.operational)
    operational_stats = {
        'operational': operational_count,
        'failed': len(satellites) - operational_count,
        'operational_rate': operational_count / len(satellites),
    }

    return {
        'battery': battery_stats,
        'storage': storage_stats,
        'operational': operational_stats,
    }
```

## Observation Validation

### Sanity Checks

```python
def validate_observation(observation):
    """Validate observation structure and values."""
    errors = []

    # Check satellite properties
    for satellite in observation.satellites:
        if not (0 <= satellite.battery <= 100):
            errors.append(f"Satellite {satellite.id} battery out of range: {satellite.battery}")
        if not (0 <= satellite.storage <= 100):
            errors.append(f"Satellite {satellite.id} storage out of range: {satellite.storage}")
        if satellite.last_action not in [None, "capture", "downlink", "maintain", "idle"]:
            errors.append(f"Satellite {satellite.id} invalid last_action: {satellite.last_action}")

    # Check weather conditions
    for region, cloud_cover in observation.weather_conditions.items():
        if not (0 <= cloud_cover <= 1):
            errors.append(f"Weather {region} cloud cover out of range: {cloud_cover}")

    # Check step count
    if observation.step_count < 0:
        errors.append(f"Negative step count: {observation.step_count}")

    return errors
```

### Observation Comparison

```python
def compare_observations(obs1, obs2):
    """Compare two observations for debugging."""
    differences = {}

    # Compare satellite states
    for i, (sat1, sat2) in enumerate(zip(obs1.satellites, obs2.satellites)):
        if sat1 != sat2:
            differences[f'satellite_{i}'] = {
                'battery': (sat1.battery, sat2.battery),
                'storage': (sat1.storage, sat2.storage),
                'position': (sat1.position, sat2.position),
                'operational': (sat1.operational, sat2.operational),
            }

    # Compare weather
    if obs1.weather_conditions != obs2.weather_conditions:
        differences['weather'] = {
            'old': obs1.weather_conditions,
            'new': obs2.weather_conditions,
        }

    # Compare step count
    if obs1.step_count != obs2.step_count:
        differences['step_count'] = (obs1.step_count, obs2.step_count)

    return differences
```

## Advanced Observation Features

### Historical Tracking

```python
class ObservationHistory:
    """Track observation history for temporal analysis."""

    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history

    def add_observation(self, observation):
        """Add new observation to history."""
        self.history.append(observation)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_trends(self):
        """Calculate trends in resource levels."""
        if len(self.history) < 2:
            return {}

        latest = self.history[-1]
        previous = self.history[-2]

        trends = {}
        for i, (latest_sat, prev_sat) in enumerate(zip(latest.satellites, previous.satellites)):
            trends[f'satellite_{i}'] = {
                'battery_change': latest_sat.battery - prev_sat.battery,
                'storage_change': latest_sat.storage - prev_sat.storage,
            }

        return trends
```

### Prediction Features

```python
def predict_future_state(current_obs, actions, steps_ahead=1):
    """Predict future observation based on planned actions."""
    # Simplified prediction model
    predicted = copy.deepcopy(current_obs)

    for _ in range(steps_ahead):
        # Apply action effects
        for sat_id, action in actions.satellite_actions.items():
            sat = predicted.satellites[sat_id]

            if action == "capture" and sat.battery >= 5 and sat.storage <= 90:
                sat.battery -= 5
                sat.storage += 10
            elif action == "downlink" and sat.battery >= 2 and sat.storage > 0:
                sat.battery -= 2
                sat.storage = max(0, sat.storage - 20)
            elif action == "maintain":
                sat.battery = min(100, sat.battery + 20)

        # Apply time-based changes
        for sat in predicted.satellites:
            sat.battery = max(0, sat.battery - 0.5)  # Battery drain

        predicted.step_count += 1

    return predicted
```

---

[← Action Space](action-space.md) | [Reward Function →](reward-function.md)