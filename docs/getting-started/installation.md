# Installation

This guide will help you install and set up the Satellite Constellation Management Environment.

## Prerequisites

- **Python**: 3.11 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: At least 4GB (8GB recommended for larger simulations)

## Installation Methods

### Option 1: Install from PyPI (Recommended)

```bash
pip install satellite-env
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/satellite-constellation-env.git
cd satellite-constellation-env

# Install in development mode
pip install -e .
```

### Option 3: Docker Installation

```bash
# Build the Docker image
docker build -t satellite-env .

# Run the container
docker run -it satellite-env
```

## Verify Installation

After installation, verify that everything works:

```python
import satellite_env

# Test basic import
from satellite_env import SatelliteConstellationEnv, Action
print("✅ Import successful")

# Test environment creation
env = SatelliteConstellationEnv()
obs = env.reset()
print(f"✅ Environment created with {len(obs.satellites)} satellites")

# Test basic step
action = Action(satellite_actions={0: "idle"})
obs, reward, done, info = env.step(action)
print(f"✅ Step executed, reward: {reward.value}")
```

## Development Dependencies

If you plan to contribute or run tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or manually install
pip install pytest black isort mypy pre-commit
```

## Troubleshooting

### Common Issues

#### Import Error
```python
# If you get import errors, check your Python path
import sys
print(sys.path)

# Try reinstalling
pip uninstall satellite-env
pip install satellite-env
```

#### Version Conflicts
```python
# Check installed versions
pip list | grep -E "(satellite-env|pydantic|numpy)"

# Use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install satellite-env
```

#### Memory Issues
If you encounter memory issues with large constellations:

```python
# Reduce satellite count for testing
env = SatelliteConstellationEnv(num_satellites=3)  # Instead of default 5

# Or use smaller tasks
from satellite_env import EasyTask
task = EasyTask()
task.setup_environment(env)
```

## Next Steps

Once installed, you can:

- [Run the Quick Start guide](quick-start.md)
- [Learn basic usage](basic-usage.md)
- [Explore the API](../api-reference/core-classes.md)

## Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the [API documentation](../api-reference/core-classes.md)
3. Open an issue on [GitHub](https://github.com/your-username/satellite-constellation-env/issues)

---

[Next: Quick Start →](quick-start.md)