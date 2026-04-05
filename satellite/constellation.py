# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Satellite Constellation Environment.

Simulates a constellation of satellites with imaging and downlinking capabilities.
"""

from typing import Dict, Any, Tuple, List
import numpy as np


class SatelliteConstellationEnv:
    """Environment for managing a satellite constellation."""

    def __init__(self, num_satellites: int = 5, max_steps: int = 100):
        """
        Initialize the satellite constellation environment.

        Args:
            num_satellites: Number of satellites in the constellation
            max_steps: Maximum steps per episode
        """
        self.num_satellites = num_satellites
        self.max_steps = max_steps
        self.current_step = 0
        self.satellites: List[Dict[str, Any]] = []
        self.ground_stations = [(0, 0), (45, 90), (-30, 120)]  # example locations
        self.weather = {"region1": 0.2, "region2": 0.5}  # cloud cover
        self.pending_tasks: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self._reset_satellites()

    def _reset_satellites(self):
        """Reset all satellites to initial state."""
        self.satellites = []
        for i in range(self.num_satellites):
            # Rough orbital positions
            pos = (
                np.random.uniform(-6371, 6371),
                np.random.uniform(-6371, 6371),
                np.random.uniform(400, 600),
            )
            self.satellites.append(
                {
                    "id": i,
                    "position": pos,
                    "battery": 100.0,
                    "storage": 0.0,
                    "last_action": "idle",
                }
            )

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment.

        Returns:
            Initial observation
        """
        self.current_step = 0
        self._reset_satellites()
        self.total_reward = 0.0
        self.pending_tasks = [
            {"type": "image_capture", "region": "region1", "priority": 1},
            {"type": "data_downlink", "station": 0, "priority": 2},
        ]
        return self._get_observation()

    def step(
        self, action: Dict[int, str]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Dictionary mapping satellite_id to action string

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1

        # Apply actions
        reward_value = 0.0
        reward_components = {}

        for sat_id, act in action.items():
            if sat_id >= len(self.satellites):
                continue
            sat = self.satellites[sat_id]

            if act == "capture":
                if sat["battery"] > 10 and sat["storage"] < 90:
                    sat["battery"] -= 5
                    sat["storage"] += 10
                    reward_value += 10  # reward for capturing
                    reward_components[f"capture_{sat_id}"] = 10
                else:
                    reward_value -= 1  # penalty for invalid action
            elif act == "downlink":
                if self._can_downlink(sat_id):
                    data_downlinked = min(sat["storage"], 20)
                    sat["storage"] -= data_downlinked
                    sat["battery"] -= 2
                    reward_value += data_downlinked * 2  # reward for downlinking
                    reward_components[f"downlink_{sat_id}"] = data_downlinked * 2
                else:
                    reward_value -= 1
            elif act == "maintain":
                sat["battery"] = min(100, sat["battery"] + 20)
                reward_value += 5  # reward for maintenance
                reward_components[f"maintain_{sat_id}"] = 5
            elif act == "idle":
                pass  # no change

            sat["last_action"] = act

        # Update positions (simple orbital motion)
        for sat in self.satellites:
            x, y, z = sat["position"]
            # Simple circular motion
            angle = np.arctan2(y, x)
            radius = np.sqrt(x**2 + y**2)
            angle += 0.01
            sat["position"] = (
                radius * np.cos(angle),
                radius * np.sin(angle),
                z,
            )

        self.total_reward += reward_value

        done = self.current_step >= self.max_steps
        obs = self._get_observation()

        return obs, reward_value, done, reward_components

    def _can_downlink(self, sat_id: int) -> bool:
        """Check if satellite can downlink to a ground station."""
        if sat_id >= len(self.satellites):
            return False
        # Simple check: always can downlink if storage > 0
        return self.satellites[sat_id]["storage"] > 0

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation."""
        return {
            "satellites": [
                {
                    "id": s["id"],
                    "position": s["position"],
                    "battery": s["battery"],
                    "storage": s["storage"],
                    "last_action": s["last_action"],
                }
                for s in self.satellites
            ],
            "time_step": self.current_step,
            "ground_stations": self.ground_stations,
            "weather_conditions": self.weather,
            "pending_tasks": self.pending_tasks,
            "total_reward": self.total_reward,
        }
