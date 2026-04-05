# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Satellite Environment.

Satellite constellation management with multiple satellites,
imaging tasks, and data downlinking capabilities.
"""

from typing import List, Dict, Any, Tuple
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


class SatelliteState(BaseModel):
    """State of an individual satellite in the constellation."""
    id: int = Field(..., description="Satellite ID")
    position: Tuple[float, float, float] = Field(..., description="Position (x, y, z) in orbit")
    battery: float = Field(..., description="Battery level 0-100")
    storage: float = Field(..., description="Storage used percentage 0-100")
    last_action: str = Field(..., description="Last action taken")


class SatelliteAction(Action):
    """Action for satellite constellation - commands for each satellite."""

    satellite_actions: Dict[int, str] = Field(
        ..., 
        description="Dictionary mapping satellite_id to action: 'capture', 'downlink', 'maintain', or 'idle'"
    )


class SatelliteObservation(Observation):
    """Observation from the Satellite environment."""

    satellites: List[SatelliteState] = Field(..., description="States of all satellites")
    time_step: int = Field(..., description="Current time step")
    ground_stations: List[Tuple[float, float]] = Field(..., description="Ground station coordinates (lat, lon)")
    weather_conditions: Dict[str, float] = Field(..., description="Cloud cover by region")
    pending_tasks: List[Dict[str, Any]] = Field(..., description="List of pending tasks")
    total_reward: float = Field(default=0.0, description="Cumulative reward")
