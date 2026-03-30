"""Top-level package for the satellite environment.

This module ensures environment variables from a local .env file are loaded
when the package is imported so scripts can rely on those variables.
"""

# Load .env early so any module import that follows can use the variables.
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	# If python-dotenv is not installed, continue without failing here.
	# Scripts will still work if env vars are provided via the environment.
	pass

from .env import SatelliteConstellationEnv, Action
from .tasks import Task, EasyTask, MediumTask, HardTask
from .graders import TaskGrader

__all__ = [
	"SatelliteConstellationEnv",
	"Action",
	"Task",
	"EasyTask",
	"MediumTask",
	"HardTask",
	"TaskGrader",
]