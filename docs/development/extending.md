# Extending Environment

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