# Task Classes

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