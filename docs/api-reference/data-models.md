# Data Models

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