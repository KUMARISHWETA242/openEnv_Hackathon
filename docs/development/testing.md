# Testing

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