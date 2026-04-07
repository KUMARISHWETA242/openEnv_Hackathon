# Quick Start

Get up and running with the Satellite Constellation Management Environment in minutes.

## Hello World Example

Here's a complete example to get you started:

```python
from satellite_env import SatelliteConstellationEnv, Action

# 1. Create the environment
env = SatelliteConstellationEnv(num_satellites=3, max_steps=10)

# 2. Reset to start a new episode
observation = env.reset()
print(f"Episode started with {len(observation.satellites)} satellites")

# 3. Run a few steps
for step in range(5):
    # Create actions for each satellite
    actions = {}
    for i, sat in enumerate(observation.satellites):
        if sat.battery > 20 and sat.storage < 80:
            actions[i] = "capture"  # Take an image
        elif sat.battery < 50:
            actions[i] = "maintain"  # Charge battery
        else:
            actions[i] = "idle"  # Do nothing

    # Execute the actions
    action = Action(satellite_actions=actions)
    observation, reward, done, info = env.step(action)

    print(f"Step {step + 1}: Reward = {reward.value:.2f}, Done = {done}")

    if done:
        break

print("Episode completed!")
```

## Running Tasks

Test your understanding with the built-in tasks:

```python
from satellite_env import EasyTask, TaskGrader

# Set up the easy task
task = EasyTask()
env = SatelliteConstellationEnv()
task.setup_environment(env)
grader = TaskGrader(task)

print(f"Task: {task.name}")
print(f"Description: {task.description}")

# Run a simple random policy
observation = env.reset()
actions_taken = []
total_reward = 0

for step in range(env.max_steps):
    # Random actions for demonstration
    actions = {i: "capture" for i in range(env.num_satellites)}
    action = Action(satellite_actions=actions)

    observation, reward, done, info = env.step(action)
    actions_taken.append(action)
    total_reward += reward.value

    if done:
        break

# Evaluate performance
final_state = env.state()
score = grader.grade_episode(env, actions_taken, final_state)

print(f"Total Reward: {total_reward:.2f}")
print(f"Task Score: {score:.3f}")
print(f"Steps Taken: {len(actions_taken)}")
```

## Using the Inference Script

For a more sophisticated agent using LLMs:

```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"

# Run inference on all tasks
python inference.py
```

This will run the environment with an LLM agent and produce scores for all three tasks.

## Visualizing Results

Monitor your agent's performance:

```python
import matplotlib.pyplot as plt

# Track metrics over time
rewards_history = []
battery_levels = []
storage_levels = []

observation = env.reset()
for step in range(50):
    # Your agent logic here
    action = Action(satellite_actions={0: "capture", 1: "maintain", 2: "idle"})
    observation, reward, done, info = env.step(action)

    # Record metrics
    rewards_history.append(reward.value)
    battery_levels.append([s.battery for s in observation.satellites])
    storage_levels.append([s.storage for s in observation.satellites])

    if done:
        break

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

ax1.plot(rewards_history)
ax1.set_title('Reward Over Time')
ax1.set_xlabel('Step')
ax1.set_ylabel('Reward')

ax2.plot(battery_levels)
ax2.set_title('Battery Levels')
ax2.set_xlabel('Step')
ax2.set_ylabel('Battery (%)')
ax2.legend([f'Satellite {i}' for i in range(len(battery_levels[0]))])

ax3.plot(storage_levels)
ax3.set_title('Storage Levels')
ax3.set_xlabel('Step')
ax3.set_ylabel('Storage (%)')
ax3.legend([f'Satellite {i}' for i in range(len(storage_levels[0]))])

plt.tight_layout()
plt.show()
```

## Next Steps

Now that you have the basics working:

1. **Learn the Details**: Understand [action and observation spaces](../user-guide/action-space.md)
2. **Explore Tasks**: Try the [medium and hard tasks](../user-guide/tasks.md)
3. **Build an Agent**: Implement your own RL agent
4. **Contribute**: Help improve the environment

## Common Patterns

### Resource-Aware Actions
```python
def smart_actions(observation):
    actions = {}
    for sat in observation.satellites:
        if sat.battery < 20:
            actions[sat.id] = "maintain"  # Critical battery
        elif sat.storage > 90:
            actions[sat.id] = "downlink"  # Full storage
        elif can_capture(sat):
            actions[sat.id] = "capture"  # Good conditions
        else:
            actions[sat.id] = "idle"     # Default
    return actions
```

### Task-Specific Strategies
```python
# For imaging-focused tasks
def imaging_strategy(observation):
    actions = {}
    for sat in observation.satellites:
        # Prioritize imaging when weather is good
        weather_good = all(w < 0.3 for w in observation.weather_conditions.values())
        if weather_good and sat.battery > 30:
            actions[sat.id] = "capture"
        else:
            actions[sat.id] = "maintain"
    return actions
```

---

[← Installation](installation.md) | [Basic Usage →](basic-usage.md)