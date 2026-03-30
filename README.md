# Satellite Constellation Management Environment

This OpenEnv environment simulates managing a constellation of satellites with limited resources to maximize mission value through reinforcement learning.

## Environment Description

Satellites have limited:
- **Battery power** 🔋 (0-100%)
- **Storage capacity** 💾 (0-100% used)
- **Communication windows** 📡 (availability to ground stations)

Tasks include:
- 🌍 **Capture Earth images** (consumes battery, fills storage)
- 📡 **Downlink data** to ground stations (requires communication window, consumes battery, frees storage)
- 🔧 **Perform maintenance maneuvers** (recharges battery)

The environment is dynamic with:
- Orbital mechanics (satellite positions change)
- Weather conditions (affects imaging quality)
- Resource depletion over time
- Ground station visibility windows

## Documentation

For detailed information, see:
- **[API Documentation](API_DOCUMENTATION.md)**: Complete API reference and usage examples
- **[Task Specifications](TASK_SPECIFICATIONS.md)**: Detailed task descriptions and evaluation criteria
- **[Environment Details](ENVIRONMENT_DETAILS.md)**: Technical implementation and simulation models
- **[Development Guide](DEVELOPMENT_GUIDE.md)**: Instructions for extending and modifying the environment

## Action and Observation Spaces

### Observation Space
The observation is a structured object containing:
- `satellites`: List of satellite states
  - `id`: Satellite identifier
  - `position`: (x, y, z) coordinates in orbit
  - `battery`: Current battery level (0-100)
  - `storage`: Storage usage percentage (0-100)
  - `last_action`: Previous action taken
- `time_step`: Current simulation time step
- `ground_stations`: List of (latitude, longitude) for ground stations
- `weather_conditions`: Dict mapping regions to cloud cover (0-1)
- `pending_tasks`: List of tasks requiring attention

### Action Space
Actions are specified per satellite as a dictionary:
- `satellite_actions`: Dict[satellite_id -> action]
- Available actions:
  - `"capture"`: Take an image (battery -5, storage +10 if possible)
  - `"downlink"`: Send data to ground station (if in range, battery -2, storage -20)
  - `"maintain"`: Perform maintenance (battery +20)
  - `"idle"`: No action

### Reward Function
Rewards provide partial progress signals:
- **+10** per successful image capture
- **+2** per unit of data downlinked
- **+5** per maintenance action
- **-1** per invalid action attempt
- Battery drain of 0.5 per time step

## Tasks

### Easy Task: Basic Imaging
- 3 satellites, 50 time steps
- Capture at least 3 images
- Maintain average battery >50%
- **Target Score**: 0.7

### Medium Task: Data Management
- 5 satellites, 100 time steps
- Capture at least 5 images
- Downlink at least 50 units of data
- Maintain average battery >30%
- **Target Score**: 0.5

### Hard Task: Constellation Coordination
- 8 satellites, 200 time steps
- Handle weather constraints
- Capture at least 10 images
- Downlink at least 100 units of data
- Maintain average battery >20%
- **Target Score**: 0.3

## Setup Instructions

### Local Installation

1. **Clone and navigate to the repository**
   ```bash
   git clone <repository-url>
   cd satellite-constellation-env
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Validate the environment**
   ```bash
   python -c "import openenv; openenv.validate('.')"
   ```

### Docker Setup

1. **Build the container**
   ```bash
   docker build -t satellite-env .
   ```

2. **Run the container**
   ```bash
   docker run -p 7860:7860 satellite-env
   ```

## Usage Example

```python
from satellite_env import SatelliteConstellationEnv, Action

# Initialize environment
env = SatelliteConstellationEnv(num_satellites=5)

# Reset for new episode
observation = env.reset()

# Take actions
action = Action(satellite_actions={0: "capture", 1: "downlink", 2: "maintain"})
observation, reward, done, info = env.step(action)

# Get current state
state = env.state()
```

## Baseline Scores

Run the baseline inference script to get reproducible scores (using Groq):

```bash
export GROQ_API_KEY="your-api-key-here"
export GROQ_API_URL="https://api.groq.ai/v1/models/<model>/completions"  # optional
python baseline.py
```

Expected baseline scores (using GPT-4):
- Easy: ~0.7
- Medium: ~0.5
- Hard: ~0.3

## Deployment to Hugging Face Spaces

1. **Create a new Space** on Hugging Face with Docker
2. **Push your code** to the repository
3. **The Space will automatically build** using the Dockerfile
4. **Access the environment** via the web interface

## Contributing

This environment is designed for the OpenEnv Hackathon. Contributions should focus on:
- Improving the physical accuracy of satellite dynamics
- Adding more realistic weather and orbital models
- Enhancing the reward function for better learning signals
- Optimizing performance for larger constellations

## License

MIT License - see LICENSE file for details.