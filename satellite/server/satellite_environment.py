# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv-facing wrapper around the canonical satellite task environment."""

from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except Exception:  # pragma: no cover - local fallback when openenv is absent
    class Environment:  # type: ignore[override]
        pass

    class State(BaseModel):
        episode_id: str = Field(default_factory=lambda: str(uuid4()))
        step_count: int = 0

try:
    from ..env import SatelliteTaskEnv
    from ..models import SatelliteAction, SatelliteObservation
except ImportError:
    from env import SatelliteTaskEnv
    from models import SatelliteAction, SatelliteObservation


class SatelliteEnvironment(Environment):
    """
    Satellite constellation management environment.

    Manages a constellation of satellites with imaging and downlinking capabilities.
    Agents must balance battery usage, storage capacity, and task completion.

    Example:
        >>> env = SatelliteEnvironment()
        >>> obs = env.reset()
        >>> print(len(obs.satellites))  # Number of satellites
        >>> 
        >>> action = SatelliteAction(satellite_actions={0: 'capture', 1: 'maintain'})
        >>> obs = env.step(action)
        >>> print(obs.total_reward)  # Accumulated reward
    """

    # Enable concurrent WebSocket sessions.
    # Each client gets their own environment instance.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        task_name: str = "medium",
        num_satellites: int = 5,
        max_steps: int = 100,
    ):
        """
        Initialize the satellite environment.

        Args:
            num_satellites: Number of satellites in the constellation
            max_steps: Maximum steps per episode
        """
        self._env = SatelliteTaskEnv(
            task_name=task_name,
            num_satellites=num_satellites, max_steps=max_steps
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> SatelliteObservation:
        """
        Reset the environment to initial state.

        Returns:
            SatelliteObservation with initial satellite states and tasks
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        observation = self._env.reset()
        self._state.episode_id = self._env.episode_id
        self._state.step_count = 0
        return observation

    def step(self, action: SatelliteAction) -> SatelliteObservation:  # type: ignore[override]
        """
        Execute one step in the environment.

        Args:
            action: SatelliteAction specifying actions for each satellite

        Returns:
            SatelliteObservation with updated states and reward
        """
        self._state.step_count += 1
        observation, _, _, _ = self._env.step(action)
        return observation

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        snapshot = self._env.state()
        self._state.episode_id = snapshot.episode_id
        self._state.step_count = snapshot.step_count
        return self._state
