# Development Guide - Satellite Constellation Management Environment

## Overview

This guide provides instructions for developers who want to extend, modify, or contribute to the Satellite Constellation Management Environment.

## Development Setup

### Prerequisites
- Python 3.11+
- Git
- Docker (for containerized testing)

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd satellite-constellation-env

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing Setup
```bash
# Run basic tests
python test_env.py

# Run inference tests (using Groq)
export GROQ_API_KEY="your-key-here"
python inference.py

# Validate OpenEnv compliance
python -c "import openenv; openenv.validate('.')"
```

## Project Structure

```
satellite-constellation-env/
├── satellite_env/           # Main package
│   ├── __init__.py         # Package exports
│   ├── env.py              # Core environment implementation
│   ├── tasks.py            # Task definitions
│   └── graders.py          # Performance evaluation
├── openenv.yaml            # Environment metadata
├── inference.py            # LLM inference script (Groq-compatible)
├── baseline.py             # Groq baseline implementation
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── README.md               # Main documentation
├── API_DOCUMENTATION.md    # API reference
├── TASK_SPECIFICATIONS.md  # Task details
├── ENVIRONMENT_DETAILS.md  # Technical implementation
└── DEVELOPMENT_GUIDE.md    # This file
```

## Extending the Environment

### Adding New Satellite Capabilities

#### 1. Extend SatelliteState Model
```python
from pydantic import BaseModel
from typing import Optional

class ExtendedSatelliteState(BaseModel):
    id: int
    position: Tuple[float, float, float]
    battery: float
    storage: float
    last_action: str
    # New fields
    temperature: float  # Operating temperature
    attitude: Tuple[float, float, float]  # Orientation
    payload_status: Dict[str, bool]  # Sensor health
```

#### 2. Update Environment Class
```python
class ExtendedSatelliteConstellationEnv(SatelliteConstellationEnv):
    def _reset_satellites(self):
        super()._reset_satellites()
        for sat in self.satellites:
            sat['temperature'] = 20.0  # Celsius
            sat['attitude'] = (0.0, 0.0, 0.0)  # Euler angles
            sat['payload_status'] = {'camera': True, 'antenna': True}
```

#### 3. Add New Actions
```python
VALID_ACTIONS = ['capture', 'downlink', 'maintain', 'idle', 'reorient', 'diagnose']

def execute_extended_action(self, satellite, action):
    if action == 'reorient':
        # Change satellite attitude
        satellite['attitude'] = self._calculate_new_attitude(satellite)
        satellite['battery'] -= 3
        return 2  # Reward for successful reorientation
    elif action == 'diagnose':
        # Run diagnostics on payload
        issues_found = self._check_payload_health(satellite)
        satellite['payload_status'].update(issues_found)
        satellite['battery'] -= 1
        return 3  # Reward for diagnostics
    else:
        return super().execute_action(satellite, action)
```

### Creating Custom Tasks

#### 1. Define Task Class
```python
from satellite_env.tasks import Task

class CustomTask(Task):
    def __init__(self):
        super().__init__(
            name="Custom: Advanced Mission",
            description="Complex mission with multiple objectives"
        )

    def setup_environment(self, env):
        # Configure environment for your task
        env.num_satellites = 6
        env.max_steps = 150

        # Set up specific initial conditions
        env.weather = {"region1": 0.3, "region2": 0.7, "region3": 0.1}
        env.pending_tasks = [
            {"type": "high_res_image", "region": "region1", "priority": 3},
            {"type": "data_downlink", "station": 0, "priority": 2},
            {"type": "maintenance", "satellite_id": 2, "priority": 1}
        ] * 8

    def get_success_criteria(self):
        return {
            "min_high_res_images": 5,
            "min_data_downlinked": 80,
            "min_satellites_healthy": 5,
            "max_steps": 150
        }
```

#### 2. Implement Custom Grader
```python
from satellite_env.graders import TaskGrader

class CustomTaskGrader(TaskGrader):
    def grade_episode(self, env, actions, final_state):
        criteria = self.task.get_success_criteria()

        # Custom evaluation logic
        high_res_captures = sum(1 for action in actions
                              for act in action.satellite_actions.values()
                              if act == 'capture_high_res')

        data_downlinked = sum(1 for action in actions
                            for act in action.satellite_actions.values()
                            if act == 'downlink')

        healthy_satellites = sum(1 for sat in final_state['satellites']
                               if sat['battery'] > 20 and all(sat['payload_status'].values()))

        # Calculate score
        score = 0.0
        score += min(1.0, high_res_captures / criteria['min_high_res_images'])
        score += min(1.0, data_downlinked / criteria['min_data_downlinked'])
        score += min(1.0, healthy_satellites / criteria['min_satellites_healthy'])

        return min(1.0, score / 3.0)
```

### Modifying Reward Functions

#### 1. Custom Reward Calculator
```python
class CustomRewardCalculator:
    def __init__(self, reward_weights=None):
        self.weights = reward_weights or {
            'capture': 10.0,
            'downlink': 2.0,
            'maintain': 5.0,
            'invalid': -1.0,
            'efficiency_bonus': 1.0,
            'time_penalty': -0.1
        }

    def calculate_reward(self, action, satellite, success, time_step):
        base_reward = 0.0

        if not success:
            return self.weights['invalid']

        if action == 'capture':
            base_reward = self.weights['capture']
            # Bonus for capturing in good conditions
            if self._good_capture_conditions(satellite):
                base_reward += self.weights['efficiency_bonus']
        elif action == 'downlink':
            data_sent = min(satellite['storage'], 20)
            base_reward = data_sent * self.weights['downlink']
        elif action == 'maintain':
            base_reward = self.weights['maintain']

        # Time-based penalties for inefficient actions
        base_reward += time_step * self.weights['time_penalty']

        return base_reward
```

#### 2. Integrate Custom Rewards
```python
class CustomSatelliteEnv(SatelliteConstellationEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_calculator = CustomRewardCalculator()

    def step(self, action):
        # ... existing step logic ...

        # Calculate custom reward
        reward_value = 0.0
        reward_components = {}

        for sat_id, act in action.satellite_actions.items():
            success, reward = self._execute_action_with_reward(sat_id, act)
            reward_value += reward
            reward_components[f'{act}_{sat_id}'] = reward

        reward = Reward(value=reward_value, components=reward_components)

        # ... rest of step logic ...
```

## Testing and Validation

### Unit Testing
```python
import unittest
from satellite_env import SatelliteConstellationEnv, Action

class TestSatelliteEnv(unittest.TestCase):
    def setUp(self):
        self.env = SatelliteConstellationEnv(num_satellites=2, max_steps=10)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(len(obs.satellites), 2)
        self.assertEqual(obs.time_step, 0)

    def test_action_execution(self):
        self.env.reset()
        action = Action(satellite_actions={0: 'capture', 1: 'idle'})
        obs, reward, done, info = self.env.step(action)

        # Verify battery decreased for satellite 0
        self.assertLess(obs.satellites[0].battery, 100)
        # Verify storage increased for satellite 0
        self.assertGreater(obs.satellites[0].storage, 0)
        # Verify reward given
        self.assertGreater(reward.value, 0)

    def test_invalid_action_penalty(self):
        # Set up satellite with low battery
        self.env.reset()
        self.env.satellites[0]['battery'] = 1  # Very low battery

        action = Action(satellite_actions={0: 'capture', 1: 'idle'})
        obs, reward, done, info = self.env.step(action)

        # Should get penalty for invalid action
        self.assertLess(reward.value, 0)
```

### Integration Testing
```python
def test_full_episode():
    env = SatelliteConstellationEnv()
    obs = env.reset()

    total_reward = 0
    steps = 0

    while not done and steps < env.max_steps:
        # Simple policy: capture when possible, maintain otherwise
        actions = {}
        for i, sat in enumerate(obs.satellites):
            if sat.battery > 10 and sat.storage < 90:
                actions[i] = 'capture'
            else:
                actions[i] = 'maintain'

        action = Action(satellite_actions=actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    print(f"Episode completed: {steps} steps, {total_reward:.2f} total reward")
    return total_reward, steps
```

### Performance Benchmarking
```python
import time

def benchmark_environment():
    env = SatelliteConstellationEnv(num_satellites=10, max_steps=1000)

    start_time = time.time()
    obs = env.reset()

    for _ in range(1000):
        action = Action(satellite_actions={i: 'idle' for i in range(10)})
        obs, reward, done, info = env.step(action)
        if done:
            break

    end_time = time.time()
    print(f"1000 steps took {end_time - start_time:.2f} seconds")
    print(f"Steps per second: {1000 / (end_time - start_time):.1f}")
```

## Debugging and Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Check Python path
import sys
print(sys.path)

# Verify package installation
import satellite_env
print(satellite_env.__file__)
```

#### 2. Action Validation Failures
```python
# Debug action validation
env = SatelliteConstellationEnv()
obs = env.reset()

print("Satellite states:")
for sat in obs.satellites:
    print(f"  ID {sat.id}: battery={sat.battery}, storage={sat.storage}")

# Test specific action
action = Action(satellite_actions={0: 'capture'})
try:
    obs, reward, done, info = env.step(action)
    print(f"Action succeeded: reward={reward.value}")
except Exception as e:
    print(f"Action failed: {e}")
```

#### 3. Reward Anomalies
```python
# Log detailed reward calculation
env = SatelliteConstellationEnv()
obs = env.reset()

action = Action(satellite_actions={0: 'capture', 1: 'downlink'})
obs, reward, done, info = env.step(action)

print(f"Total reward: {reward.value}")
print("Reward components:")
for component, value in reward.components.items():
    print(f"  {component}: {value}")
```

### Logging and Monitoring
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebugSatelliteEnv(SatelliteConstellationEnv):
    def step(self, action):
        logger.debug(f"Executing action: {action}")
        result = super().step(action)
        obs, reward, done, info = result
        logger.debug(f"Reward: {reward.value}, Done: {done}")
        return result
```

## Deployment and Distribution

### Docker Development
```dockerfile
# Use multi-stage build for development
FROM python:3.11-slim as base

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM base as development
COPY . .
RUN pip install -e .
CMD ["python", "test_env.py"]

FROM base as production
COPY . .
RUN pip install .
EXPOSE 7860
CMD ["python", "-c", "import satellite_env; print('Environment ready')"]
```

### Hugging Face Spaces Deployment
```python
# app.py for HF Spaces
import gradio as gr
from satellite_env import SatelliteConstellationEnv, Action

def create_demo():
    env = SatelliteConstellationEnv(num_satellites=3)

    def step_simulation(action_dict):
        action = Action(satellite_actions=action_dict)
        obs, reward, done, info = env.step(action)

        status = f"Reward: {reward.value:.2f}, Done: {done}\n"
        for sat in obs.satellites:
            status += f"Sat {sat.id}: Battery {sat.battery:.1f}%, Storage {sat.storage:.1f}%\n"

        return status

    # Create Gradio interface
    # ... interface definition ...

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for all function parameters and return values
- Write comprehensive docstrings
- Keep functions focused and modular

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for complex features
- Performance benchmarks for optimizations
- Documentation updates for API changes

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request with detailed description

### Issue Reporting
- Use GitHub issues for bug reports and feature requests
- Include environment details and reproduction steps
- Provide example code when possible

## Performance Optimization

### Profiling
```python
import cProfile
import pstats

def profile_environment():
    env = SatelliteConstellationEnv(num_satellites=50)

    profiler = cProfile.Profile()
    profiler.enable()

    obs = env.reset()
    for _ in range(100):
        action = Action(satellite_actions={i: 'idle' for i in range(50)})
        obs, reward, done, info = env.step(action)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Optimization Techniques
1. **Vectorization**: Use NumPy for batch operations
2. **Caching**: Cache expensive calculations
3. **Lazy Evaluation**: Compute values only when needed
4. **Memory Pooling**: Reuse object instances

### Scaling Considerations
- Environment scales linearly with satellite count
- Memory usage grows with task queue size
- Consider distributed execution for very large constellations

## Future Development Roadmap

### Short Term (1-3 months)
- [ ] Add 3D orbital mechanics
- [ ] Implement real weather data integration
- [ ] Create web-based visualization interface
- [ ] Add more task types and complexity levels

### Medium Term (3-6 months)
- [ ] Multi-agent RL support
- [ ] Inter-satellite communication
- [ ] Failure mode simulation
- [ ] Performance optimization for large-scale simulations

### Long Term (6+ months)
- [ ] Real satellite telemetry integration
- [ ] Machine learning-based weather prediction
- [ ] Autonomous mission planning
- [ ] Multi-objective optimization

This development guide provides a foundation for extending and improving the Satellite Constellation Management Environment. Contributions that follow these guidelines will help advance the field of autonomous satellite operations.