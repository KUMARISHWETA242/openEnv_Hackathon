# Setup

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