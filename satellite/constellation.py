"""Deterministic satellite constellation simulator."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


VALID_ACTIONS = {"capture", "downlink", "maintain", "idle"}


class SatelliteConstellationEnv:
    """Environment for managing a satellite constellation."""

    def __init__(self, num_satellites: int = 5, max_steps: int = 100, seed: int = 7):
        self.num_satellites = num_satellites
        self.max_steps = max_steps
        self.seed = seed
        self.episode_index = 0
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.satellites: List[Dict[str, Any]] = []
        self.ground_stations = [(0, 0), (45, 90), (-30, 120)]
        self.weather = {"region1": 0.2, "region2": 0.5}
        self.pending_tasks: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.metrics: Dict[str, Any] = {}
        self.action_trace: List[Dict[str, Any]] = []
        self._reset_metrics()
        self._reset_satellites()

    def _reset_rng(self) -> None:
        self.rng = np.random.default_rng(self.seed + self.episode_index)

    def _reset_metrics(self) -> None:
        self.metrics = {
            "successful_captures": 0,
            "capture_task_completions": 0,
            "downlink_units": 0.0,
            "downlink_task_completions": 0,
            "invalid_actions": 0,
            "idle_steps": 0,
            "maintain_actions": 0,
            "repeated_action_penalties": 0,
            "destructive_action_penalties": 0,
            "tasks_completed": 0,
        }
        self.action_trace = []

    def _reset_satellites(self) -> None:
        self.satellites = []
        for i in range(self.num_satellites):
            pos = (
                float(self.rng.uniform(-6371, 6371)),
                float(self.rng.uniform(-6371, 6371)),
                float(self.rng.uniform(400, 600)),
            )
            self.satellites.append(
                {
                    "id": i,
                    "position": pos,
                    "battery": 100.0,
                    "storage": 0.0,
                    "last_action": "idle",
                    "repeat_count": 0,
                }
            )

    def reset(self) -> Dict[str, Any]:
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_index += 1
        self._reset_rng()
        self._reset_metrics()
        self._reset_satellites()
        if not self.pending_tasks:
            self.pending_tasks = [
                {"id": "img-1", "type": "image_capture", "region": "region1", "priority": 1},
                {
                    "id": "down-1",
                    "type": "data_downlink",
                    "station": 0,
                    "priority": 2,
                    "units_remaining": 20,
                },
            ]
        else:
            self.pending_tasks = [self._clone_task(task) for task in self.pending_tasks]
        return self._get_observation()

    def step(
        self, action: Dict[int, str]
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        reward_value = 0.0
        reward_components: Dict[str, float] = {}

        for sat in self.satellites:
            sat_id = sat["id"]
            act = action.get(sat_id, "idle")
            if act not in VALID_ACTIONS:
                act = "idle"
                reward_value -= 1.0
                self.metrics["invalid_actions"] += 1
                reward_components[f"invalid_{sat_id}"] = reward_components.get(f"invalid_{sat_id}", 0.0) - 1.0

            sat_reward, sat_components = self._apply_action(sat, act)
            reward_value += sat_reward
            for name, value in sat_components.items():
                reward_components[name] = reward_components.get(name, 0.0) + value

        self._advance_positions()
        reward_value += self._apply_passive_dynamics(reward_components)
        self.total_reward += reward_value
        done = self.current_step >= self.max_steps or all(s["battery"] <= 0 for s in self.satellites)
        observation = self._get_observation()
        info = {
            "reward_components": reward_components,
            "metrics": dict(self.metrics),
            "tasks_remaining": len(self.pending_tasks),
            "seed": self.seed + self.episode_index,
            "step": self.current_step,
        }
        self.action_trace.append(
            {
                "step": self.current_step,
                "action": dict(action),
                "reward": reward_value,
                "reward_components": dict(reward_components),
                "metrics": dict(self.metrics),
            }
        )
        return observation, reward_value, done, info

    def _apply_action(self, sat: Dict[str, Any], action: str) -> Tuple[float, Dict[str, float]]:
        reward = 0.0
        components: Dict[str, float] = {}
        sat_id = sat["id"]

        if action == sat["last_action"]:
            sat["repeat_count"] += 1
        else:
            sat["repeat_count"] = 0

        if sat["repeat_count"] >= 2 and action != "idle":
            reward -= 1.0
            self.metrics["repeated_action_penalties"] += 1
            components[f"repeat_penalty_{sat_id}"] = -1.0

        if action == "capture":
            capture_reward = self._handle_capture(sat)
            reward += capture_reward
            components[f"capture_{sat_id}"] = capture_reward
        elif action == "downlink":
            downlink_reward = self._handle_downlink(sat)
            reward += downlink_reward
            components[f"downlink_{sat_id}"] = downlink_reward
        elif action == "maintain":
            maintain_reward = self._handle_maintain(sat)
            reward += maintain_reward
            components[f"maintain_{sat_id}"] = maintain_reward
        else:
            idle_reward = self._handle_idle(sat)
            reward += idle_reward
            components[f"idle_{sat_id}"] = idle_reward

        sat["last_action"] = action
        return reward, {k: v for k, v in components.items() if abs(v) > 1e-9}

    def _handle_capture(self, sat: Dict[str, Any]) -> float:
        sat_id = sat["id"]
        if sat["battery"] <= 12 or sat["storage"] >= 90:
            self.metrics["invalid_actions"] += 1
            self.metrics["destructive_action_penalties"] += 1
            return -3.0

        task = self._next_task("image_capture")
        if task is None:
            sat["battery"] = max(0.0, sat["battery"] - 4.0)
            sat["storage"] = min(100.0, sat["storage"] + 8.0)
            self.metrics["destructive_action_penalties"] += 1
            return -2.0

        cloud_cover = float(self.weather.get(task["region"], 0.5))
        task_bonus = max(1.0, 4.0 * (1.0 - cloud_cover))
        priority_bonus = float(task.get("priority", 1))
        sat["battery"] = max(0.0, sat["battery"] - 5.0)
        sat["storage"] = min(100.0, sat["storage"] + 10.0)
        self.metrics["successful_captures"] += 1
        self.metrics["capture_task_completions"] += 1
        self._complete_task(task["id"])
        return 6.0 + task_bonus + priority_bonus

    def _handle_downlink(self, sat: Dict[str, Any]) -> float:
        if sat["battery"] <= 5 or sat["storage"] <= 0 or not self._can_downlink(sat["id"]):
            self.metrics["invalid_actions"] += 1
            return -2.0

        task = self._next_task("data_downlink")
        if task is None:
            sent = min(sat["storage"], 10.0)
            sat["storage"] -= sent
            sat["battery"] = max(0.0, sat["battery"] - 2.0)
            self.metrics["destructive_action_penalties"] += 1
            return -1.0

        units_remaining = float(task.get("units_remaining", 20.0))
        sent = min(sat["storage"], 20.0, units_remaining)
        if sent <= 0:
            self.metrics["invalid_actions"] += 1
            return -1.0

        sat["storage"] -= sent
        sat["battery"] = max(0.0, sat["battery"] - 2.0)
        task["units_remaining"] = max(0.0, units_remaining - sent)
        self.metrics["downlink_units"] += sent

        reward = sent * 1.4 + float(task.get("priority", 1))
        if task["units_remaining"] <= 0:
            self.metrics["downlink_task_completions"] += 1
            self._complete_task(task["id"])
            reward += 4.0

        return reward

    def _handle_maintain(self, sat: Dict[str, Any]) -> float:
        self.metrics["maintain_actions"] += 1
        battery_before = sat["battery"]
        sat["battery"] = min(100.0, sat["battery"] + 18.0)
        if battery_before < 35.0:
            return 3.0
        if battery_before < 60.0:
            return 1.0
        self.metrics["destructive_action_penalties"] += 1
        return -1.0

    def _handle_idle(self, sat: Dict[str, Any]) -> float:
        self.metrics["idle_steps"] += 1
        if self.pending_tasks and sat["battery"] > 30 and sat["storage"] < 80:
            return -0.5
        if sat["battery"] < 20:
            sat["battery"] = min(100.0, sat["battery"] + 1.0)
            return 0.2
        return -0.1

    def _advance_positions(self) -> None:
        for sat in self.satellites:
            x, y, z = sat["position"]
            angle = np.arctan2(y, x)
            radius = np.sqrt(x**2 + y**2)
            angle += 0.01
            sat["position"] = (
                float(radius * np.cos(angle)),
                float(radius * np.sin(angle)),
                float(z),
            )

    def _apply_passive_dynamics(self, reward_components: Dict[str, float]) -> float:
        battery_penalty = 0.0
        for sat in self.satellites:
            sat["battery"] = max(0.0, sat["battery"] - 0.5)
            if sat["battery"] < 10.0:
                battery_penalty -= 0.75
            if sat["storage"] > 95.0:
                battery_penalty -= 0.5
        if battery_penalty:
            reward_components["resource_risk"] = reward_components.get("resource_risk", 0.0) + battery_penalty
        return battery_penalty

    def _next_task(self, task_type: str) -> Optional[Dict[str, Any]]:
        matches = [task for task in self.pending_tasks if task["type"] == task_type]
        if not matches:
            return None
        matches.sort(key=lambda task: (-int(task.get("priority", 1)), str(task["id"])))
        return matches[0]

    def _complete_task(self, task_id: str) -> None:
        remaining: List[Dict[str, Any]] = []
        completed = False
        for task in self.pending_tasks:
            if not completed and task["id"] == task_id:
                completed = True
                continue
            remaining.append(task)
        if completed:
            self.pending_tasks = remaining
            self.metrics["tasks_completed"] += 1

    def _can_downlink(self, sat_id: int) -> bool:
        if sat_id >= len(self.satellites):
            return False
        return self.satellites[sat_id]["storage"] > 0

    def _clone_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in task.items()}

    def _get_observation(self) -> Dict[str, Any]:
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
            "pending_tasks": [self._clone_task(task) for task in self.pending_tasks],
            "total_reward": self.total_reward,
        }
