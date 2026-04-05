# Examples

## Usage Examples

### Basic Environment Interaction
```python
from satellite_env import SatelliteConstellationEnv, Action

env = SatelliteConstellationEnv(num_satellites=3)
obs = env.reset()

action = Action(satellite_actions={0: "capture", 1: "maintain", 2: "idle"})
obs, reward, done, info = env.step(action)
```

### Running Tasks
```python
from satellite_env import EasyTask, TaskGrader

task = EasyTask()
env = SatelliteConstellationEnv()
task.setup_environment(env)

grader = TaskGrader(task)
# ... run episode ...
score = grader.grade_episode(env, actions, final_state)
```

## Extension Points

### Custom Tasks
Create new tasks by inheriting from `Task`:

```python
class CustomTask(Task):
    def setup_environment(self, env):
        # Configure environment
        pass

    def get_success_criteria(self):
        return {"custom_metric": threshold}
```

### Environment Modifications
Extend `SatelliteConstellationEnv` to add:
- More realistic orbital mechanics
- Additional satellite capabilities
- Complex weather models
- Communication protocols

## Error Handling

The environment includes robust error handling:
- Invalid actions default to 'idle'
- Resource constraints prevent impossible operations
- Graceful degradation on boundary conditions

## Performance Considerations

- Environment is lightweight and runs efficiently
- Scales to 100+ satellites for large-scale simulations
- Minimal dependencies for easy deployment