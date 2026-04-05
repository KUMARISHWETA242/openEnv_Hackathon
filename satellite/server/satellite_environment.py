# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Satellite Environment Implementation.

Manages a constellation of satellites for imaging and data downlinking tasks.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SatelliteAction, SatelliteObservation, SatelliteState
    from ..constellation import SatelliteConstellationEnv
except ImportError:
    from models import SatelliteAction, SatelliteObservation, SatelliteState
    from constellation import SatelliteConstellationEnv


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

    def __init__(self, num_satellites: int = 5, max_steps: int = 100):
        """
        Initialize the satellite environment.

        Args:
            num_satellites: Number of satellites in the constellation
            max_steps: Maximum steps per episode
        """
        self._constellation = SatelliteConstellationEnv(
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
        obs_dict = self._constellation.reset()

        return self._dict_to_observation(obs_dict, reward=0.0, done=False)

    def step(self, action: SatelliteAction) -> SatelliteObservation:  # type: ignore[override]
        """
        Execute one step in the environment.

        Args:
            action: SatelliteAction specifying actions for each satellite

        Returns:
            SatelliteObservation with updated states and reward
        """
        self._state.step_count += 1

        obs_dict, reward, done, info = self._constellation.step(
            action.satellite_actions
        )

        return self._dict_to_observation(obs_dict, reward=reward, done=done, info=info)

    def _dict_to_observation(
        self, obs_dict, reward: float, done: bool, info: dict = None
    ) -> SatelliteObservation:
        """Convert internal observation dict to SatelliteObservation."""
        satellites = [SatelliteState(**sat) for sat in obs_dict["satellites"]]

        return SatelliteObservation(
            satellites=satellites,
            time_step=obs_dict["time_step"],
            ground_stations=obs_dict["ground_stations"],
            weather_conditions=obs_dict["weather_conditions"],
            pending_tasks=obs_dict["pending_tasks"],
            total_reward=obs_dict["total_reward"],
            done=done,
            reward=reward,
            metadata=info or {},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
