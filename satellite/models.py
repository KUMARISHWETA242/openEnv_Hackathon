# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the satellite environment."""

from typing import Any, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation
except Exception:  # pragma: no cover - local simulator fallback
    Action = BaseModel
    Observation = BaseModel


class SatelliteState(BaseModel):
    """State of an individual satellite in the constellation."""

    id: int = Field(..., description="Satellite ID")
    position: Tuple[float, float, float] = Field(
        ..., description="Position (x, y, z) in orbit"
    )
    battery: float = Field(..., description="Battery level 0-100")
    storage: float = Field(..., description="Storage used percentage 0-100")
    last_action: str = Field(..., description="Last action taken")


class SatelliteAction(Action):
    """Action for satellite constellation - commands for each satellite."""

    satellite_actions: Dict[int, Literal["capture", "downlink", "maintain", "idle"]] = Field(
        ...,
        description=(
            "Dictionary mapping satellite_id to action: "
            "'capture', 'downlink', 'maintain', or 'idle'"
        ),
    )


class SatelliteObservation(Observation):
    """Observation from the Satellite environment."""

    satellites: List[SatelliteState] = Field(..., description="States of all satellites")
    time_step: int = Field(..., description="Current time step")
    ground_stations: List[Tuple[float, float]] = Field(
        ..., description="Ground station coordinates (lat, lon)"
    )
    weather_conditions: Dict[str, float] = Field(..., description="Cloud cover by region")
    pending_tasks: List[Dict[str, Any]] = Field(..., description="List of pending tasks")
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: float = Field(default=0.0, description="Immediate reward from the latest step")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional step metadata"
    )


class SatelliteReward(BaseModel):
    """Typed reward model returned by the canonical environment API."""

    value: float = Field(..., description="Scalar reward for the latest step")
    components: Dict[str, float] = Field(
        default_factory=dict, description="Reward component breakdown"
    )


class SatelliteEnvState(BaseModel):
    """Typed state snapshot returned by state()."""

    episode_id: str = Field(..., description="Unique episode identifier")
    task_name: str = Field(..., description="Active task preset")
    step_count: int = Field(..., description="Current environment step count")
    max_steps: int = Field(..., description="Maximum steps allowed in the episode")
    seed: int = Field(..., description="Deterministic seed for the current task setup")
    done: bool = Field(..., description="Whether the episode is finished")
    total_reward: float = Field(..., description="Accumulated reward")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Deterministic episode metrics used for grading"
    )
    satellites: List[SatelliteState] = Field(..., description="Current satellite states")
    ground_stations: List[Tuple[float, float]] = Field(
        ..., description="Ground station coordinates (lat, lon)"
    )
    weather_conditions: Dict[str, float] = Field(..., description="Cloud cover by region")
    pending_tasks: List[Dict[str, Any]] = Field(..., description="Remaining visible tasks")
