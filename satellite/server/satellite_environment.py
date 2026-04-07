# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv-facing wrapper around the canonical satellite task environment."""

from __future__ import annotations

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

import importlib

# Prefer absolute imports when this package is placed on PYTHONPATH as a
# flat layout (e.g., /app/env is on PYTHONPATH in the Docker image).
SatelliteTaskEnv = None
SatelliteAction = None
SatelliteObservation = None

try:
    from env import SatelliteTaskEnv
    from models import SatelliteAction, SatelliteObservation
except Exception:
    # Try named package imports (e.g., when installed as `satellite` or `env`).
    for pkg in ("satellite", "env"):
        try:
            mod_env = importlib.import_module(f"{pkg}.env")
            mod_models = importlib.import_module(f"{pkg}.models")
            SatelliteTaskEnv = getattr(mod_env, "SatelliteTaskEnv")
            SatelliteAction = getattr(mod_models, "SatelliteAction")
            SatelliteObservation = getattr(mod_models, "SatelliteObservation")
            break
        except Exception:
            continue

    if SatelliteTaskEnv is None:
        # Last resort: load the sibling files directly from disk.
        try:
            import importlib.util
            from pathlib import Path

            here = Path(__file__).resolve().parent
            pkg_root = here.parent
            env_path = pkg_root / "env.py"
            models_path = pkg_root / "models.py"

            if env_path.exists() and models_path.exists():
                spec_env = importlib.util.spec_from_file_location(
                    "satellite_env_fallback", str(env_path)
                )
                mod_env = importlib.util.module_from_spec(spec_env)  # type: ignore[arg-type]
                spec_env.loader.exec_module(mod_env)  # type: ignore[attr-defined]

                spec_models = importlib.util.spec_from_file_location(
                    "satellite_models_fallback", str(models_path)
                )
                mod_models = importlib.util.module_from_spec(spec_models)  # type: ignore[arg-type]
                spec_models.loader.exec_module(mod_models)  # type: ignore[attr-defined]

                SatelliteTaskEnv = getattr(mod_env, "SatelliteTaskEnv")
                SatelliteAction = getattr(mod_models, "SatelliteAction")
                SatelliteObservation = getattr(mod_models, "SatelliteObservation")
            else:
                raise
        except Exception:
            raise


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

    def reset(self) -> "SatelliteObservation":
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

    def step(self, action: "SatelliteAction") -> "SatelliteObservation":  # type: ignore[override]
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
