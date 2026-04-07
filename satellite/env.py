"""Canonical local environment API for the satellite submission."""

from typing import Any, Dict, Tuple
from uuid import uuid4

from .constellation import SatelliteConstellationEnv
from .models import (
    SatelliteAction,
    SatelliteEnvState,
    SatelliteObservation,
    SatelliteReward,
    SatelliteState,
)
from .tasks import EasyTask, HardTask, MediumTask, Task


TASK_PRESETS = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}


class SatelliteTaskEnv:
    """Canonical typed environment with reset/step/state methods."""

    def __init__(
        self,
        task_name: str = "medium",
        num_satellites: int | None = None,
        max_steps: int | None = None,
    ):
        if task_name not in TASK_PRESETS:
            raise ValueError(
                f"Unsupported task_name '{task_name}'. Expected one of {sorted(TASK_PRESETS)}."
            )

        self.task_name = task_name
        self.task: Task = TASK_PRESETS[task_name]()
        default_satellites = num_satellites or 5
        default_max_steps = max_steps or 100
        self._sim = SatelliteConstellationEnv(
            num_satellites=default_satellites,
            max_steps=default_max_steps,
        )
        self.episode_id = str(uuid4())
        self.done = False
        self.last_info: Dict[str, Any] = {}
        self.last_reward = SatelliteReward(value=0.0, components={})

        self.task.setup_environment(self._sim)

    @staticmethod
    def list_tasks() -> Dict[str, str]:
        """List selectable task presets."""
        return {name: task_cls().description for name, task_cls in TASK_PRESETS.items()}

    def set_task(self, task_name: str) -> None:
        """Switch to a different task preset."""
        if task_name not in TASK_PRESETS:
            raise ValueError(
                f"Unsupported task_name '{task_name}'. Expected one of {sorted(TASK_PRESETS)}."
            )
        self.task_name = task_name
        self.task = TASK_PRESETS[task_name]()
        self.task.setup_environment(self._sim)

    def reset(self) -> SatelliteObservation:
        """Reset the active task and return the initial observation."""
        self.episode_id = str(uuid4())
        self.done = False
        self.last_info = {"task_name": self.task_name, "reset": True}
        self.last_reward = SatelliteReward(value=0.0, components={})
        self.task.setup_environment(self._sim)
        observation_dict = self._sim.reset()
        return self._observation_from_dict(
            observation_dict,
            reward=self.last_reward.value,
            done=self.done,
            metadata=self.last_info,
        )

    def step(
        self, action: SatelliteAction
    ) -> Tuple[SatelliteObservation, SatelliteReward, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info)."""
        observation_dict, reward_value, done, info = self._sim.step(action.satellite_actions)
        self.done = done
        self.last_info = {"task_name": self.task_name, **info}
        self.last_reward = SatelliteReward(
            value=reward_value,
            components=info.get("reward_components", {}),
        )
        observation = self._observation_from_dict(
            observation_dict,
            reward=reward_value,
            done=done,
            metadata=self.last_info,
        )
        return observation, self.last_reward, done, self.last_info

    def state(self) -> SatelliteEnvState:
        """Return a typed state snapshot for the current episode."""
        observation_dict = self._sim._get_observation()
        satellites = [SatelliteState(**sat) for sat in observation_dict["satellites"]]
        return SatelliteEnvState(
            episode_id=self.episode_id,
            task_name=self.task_name,
            step_count=self._sim.current_step,
            max_steps=self._sim.max_steps,
            seed=self._sim.seed + self._sim.episode_index,
            done=self.done,
            total_reward=observation_dict["total_reward"],
            metrics={key: float(value) for key, value in self._sim.metrics.items()},
            satellites=satellites,
            ground_stations=observation_dict["ground_stations"],
            weather_conditions=observation_dict["weather_conditions"],
            pending_tasks=observation_dict["pending_tasks"],
        )

    def _observation_from_dict(
        self,
        observation_dict: Dict[str, Any],
        reward: float,
        done: bool,
        metadata: Dict[str, Any],
    ) -> SatelliteObservation:
        satellites = [SatelliteState(**sat) for sat in observation_dict["satellites"]]
        return SatelliteObservation(
            satellites=satellites,
            time_step=observation_dict["time_step"],
            ground_stations=observation_dict["ground_stations"],
            weather_conditions=observation_dict["weather_conditions"],
            pending_tasks=observation_dict["pending_tasks"],
            total_reward=observation_dict["total_reward"],
            reward=reward,
            done=done,
            metadata=metadata,
        )
