"""Task presets for the satellite constellation simulator."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .constellation import SatelliteConstellationEnv


class Task(ABC):
    """Base task preset."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def setup_environment(self, env: SatelliteConstellationEnv) -> None:
        """Apply task-specific configuration to the environment."""

    @abstractmethod
    def get_success_criteria(self) -> Dict[str, Any]:
        """Return deterministic task success criteria."""


class EasyTask(Task):
    def __init__(self):
        super().__init__(
            "Easy: Basic Imaging",
            "Capture images with 3 satellites while keeping the fleet healthy.",
        )

    def setup_environment(self, env: SatelliteConstellationEnv) -> None:
        env.num_satellites = 3
        env.max_steps = 50
        env.seed = 101
        env._reset_satellites()
        env.weather = {"region1": 0.2, "region2": 0.5}
        env.pending_tasks = [
            {
                "id": f"easy-img-{idx}",
                "type": "image_capture",
                "region": "region1",
                "priority": 1,
            }
            for idx in range(1, 6)
        ]

    def get_success_criteria(self) -> Dict[str, Any]:
        return {
            "min_images_captured": 3,
            "min_tasks_completed": 3,
            "min_battery_final": 50,
            "max_invalid_actions": 3,
            "max_steps": 50,
        }


class MediumTask(Task):
    def __init__(self):
        super().__init__(
            "Medium: Data Management",
            "Capture images and downlink data with 5 satellites while balancing resources.",
        )

    def setup_environment(self, env: SatelliteConstellationEnv) -> None:
        env.num_satellites = 5
        env.max_steps = 100
        env.seed = 202
        env._reset_satellites()
        env.weather = {"region1": 0.2, "region2": 0.5}
        env.pending_tasks = (
            [
                {
                    "id": f"med-img-{idx}",
                    "type": "image_capture",
                    "region": "region1",
                    "priority": 1,
                }
                for idx in range(1, 7)
            ]
            + [
                {
                    "id": f"med-down-{idx}",
                    "type": "data_downlink",
                    "station": idx % 2,
                    "priority": 2,
                    "units_remaining": 10,
                }
                for idx in range(1, 7)
            ]
        )

    def get_success_criteria(self) -> Dict[str, Any]:
        return {
            "min_images_captured": 5,
            "min_data_downlinked": 50,
            "min_tasks_completed": 8,
            "min_battery_final": 30,
            "max_invalid_actions": 8,
            "max_steps": 100,
        }


class HardTask(Task):
    def __init__(self):
        super().__init__(
            "Hard: Constellation Coordination",
            "Manage 8 satellites under heavier weather and task pressure.",
        )

    def setup_environment(self, env: SatelliteConstellationEnv) -> None:
        env.num_satellites = 8
        env.max_steps = 200
        env.seed = 303
        env._reset_satellites()
        env.weather = {"region1": 0.8, "region2": 0.3, "region3": 0.6}
        env.pending_tasks = (
            [
                {
                    "id": f"hard-img-r1-{idx}",
                    "type": "image_capture",
                    "region": "region1",
                    "priority": 1,
                }
                for idx in range(1, 7)
            ]
            + [
                {
                    "id": f"hard-img-r2-{idx}",
                    "type": "image_capture",
                    "region": "region2",
                    "priority": 2,
                }
                for idx in range(1, 7)
            ]
            + [
                {
                    "id": f"hard-down-{idx}",
                    "type": "data_downlink",
                    "station": idx % 3,
                    "priority": 2,
                    "units_remaining": 12.5,
                }
                for idx in range(1, 9)
            ]
        )

    def get_success_criteria(self) -> Dict[str, Any]:
        return {
            "min_images_captured": 10,
            "min_data_downlinked": 100,
            "min_tasks_completed": 14,
            "min_battery_final": 20,
            "max_invalid_actions": 12,
            "max_steps": 200,
        }
