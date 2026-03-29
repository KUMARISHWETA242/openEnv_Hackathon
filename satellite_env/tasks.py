from typing import Dict, Any
from abc import ABC, abstractmethod
from .env import SatelliteConstellationEnv

class Task(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def setup_environment(self, env: SatelliteConstellationEnv):
        """Set up the environment for this task"""
        pass

    @abstractmethod
    def get_success_criteria(self) -> Dict[str, Any]:
        """Return criteria for success"""
        pass

class EasyTask(Task):
    def __init__(self):
        super().__init__(
            "Easy: Basic Imaging",
            "Capture images with 3 satellites, maintain battery above 50%"
        )

    def setup_environment(self, env: SatelliteConstellationEnv):
        env.num_satellites = 3
        env.max_steps = 50
        env._reset_satellites()
        env.pending_tasks = [
            {'type': 'image_capture', 'region': 'region1', 'priority': 1}
        ] * 5

    def get_success_criteria(self) -> Dict[str, Any]:
        return {
            'min_images_captured': 3,
            'min_battery_final': 50,
            'max_steps': 50
        }

class MediumTask(Task):
    def __init__(self):
        super().__init__(
            "Medium: Data Management",
            "Capture images and downlink data with 5 satellites, balance resources"
        )

    def setup_environment(self, env: SatelliteConstellationEnv):
        env.num_satellites = 5
        env.max_steps = 100
        env._reset_satellites()
        env.pending_tasks = [
            {'type': 'image_capture', 'region': 'region1', 'priority': 1},
            {'type': 'data_downlink', 'station': 0, 'priority': 2}
        ] * 10

    def get_success_criteria(self) -> Dict[str, Any]:
        return {
            'min_images_captured': 5,
            'min_data_downlinked': 50,
            'min_battery_final': 30,
            'max_steps': 100
        }

class HardTask(Task):
    def __init__(self):
        super().__init__(
            "Hard: Constellation Coordination",
            "Manage 8 satellites with weather constraints and multiple tasks"
        )

    def setup_environment(self, env: SatelliteConstellationEnv):
        env.num_satellites = 8
        env.max_steps = 200
        env._reset_satellites()
        env.weather = {"region1": 0.8, "region2": 0.3, "region3": 0.6}  # more cloudy
        env.pending_tasks = [
            {'type': 'image_capture', 'region': 'region1', 'priority': 1},
            {'type': 'image_capture', 'region': 'region2', 'priority': 1},
            {'type': 'data_downlink', 'station': 0, 'priority': 2},
            {'type': 'data_downlink', 'station': 1, 'priority': 2}
        ] * 20

    def get_success_criteria(self) -> Dict[str, Any]:
        return {
            'min_images_captured': 10,
            'min_data_downlinked': 100,
            'min_battery_final': 20,
            'max_steps': 200
        }