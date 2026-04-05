# Core Classes

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