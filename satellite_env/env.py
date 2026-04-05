from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
import numpy as np
from dataclasses import dataclass

class SatelliteState(BaseModel):
    id: int
    position: Tuple[float, float, float]  # x, y, z in orbit
    battery: float  # 0-100
    storage: float  # 0-100 (percentage used)
    last_action: str

class Observation(BaseModel):
    satellites: List[SatelliteState]
    time_step: int
    ground_stations: List[Tuple[float, float]]  # lat, lon
    weather_conditions: Dict[str, float]  # region -> cloud cover
    pending_tasks: List[Dict[str, Any]]  # tasks to be assigned

class Action(BaseModel):
    satellite_actions: Dict[int, str]  # satellite_id -> action ('capture', 'downlink', 'maintain', 'idle')

class Reward(BaseModel):
    value: float
    components: Dict[str, float]  # breakdown of reward

class SatelliteConstellationEnv:
    def __init__(self, num_satellites: int = 5, max_steps: int = 100):
        self.num_satellites = num_satellites
        self.max_steps = max_steps
        self.current_step = 0
        self.satellites = []
        self.ground_stations = [(0, 0), (45, 90), (-30, 120)]  # example locations
        self.weather = {"region1": 0.2, "region2": 0.5}  # cloud cover
        self.pending_tasks = []
        self._reset_satellites()

    def _reset_satellites(self):
        self.satellites = []
        for i in range(self.num_satellites):
            pos = (np.random.uniform(-6371, 6371), np.random.uniform(-6371, 6371), np.random.uniform(400, 600))  # rough orbit
            self.satellites.append({
                'id': i,
                'position': pos,
                'battery': 100.0,
                'storage': 0.0,
                'last_action': 'idle'
            })

    def reset(self) -> Observation:
        self.current_step = 0
        self._reset_satellites()
        self.pending_tasks = [
            {'type': 'image_capture', 'region': 'region1', 'priority': 1},
            {'type': 'data_downlink', 'station': 0, 'priority': 2}
        ]
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.current_step += 1

        # Apply actions
        reward_value = 0.0
        reward_components = {}

        for sat_id, act in action.satellite_actions.items():
            if sat_id >= len(self.satellites):
                continue
            sat = self.satellites[sat_id]

            if act == 'capture':
                if sat['battery'] > 10 and sat['storage'] < 90:
                    sat['battery'] -= 5
                    sat['storage'] += 10
                    reward_value += 10  # reward for capturing
                    reward_components[f'capture_{sat_id}'] = 10
                else:
                    reward_value -= 1  # penalty for invalid action
            elif act == 'downlink':
                if self._can_downlink(sat_id):
                    data_downlinked = min(sat['storage'], 20)
                    sat['storage'] -= data_downlinked
                    sat['battery'] -= 2
                    reward_value += data_downlinked * 2  # reward for downlinking
                    reward_components[f'downlink_{sat_id}'] = data_downlinked * 2
                else:
                    reward_value -= 1
            elif act == 'maintain':
                sat['battery'] = min(100, sat['battery'] + 20)
                reward_value += 5  # reward for maintenance
                reward_components[f'maintain_{sat_id}'] = 5
            elif act == 'idle':
                pass  # no change

            sat['last_action'] = act

        # Update positions (simple simulation)
        for sat in self.satellites:
            # Simple orbital motion
            sat['position'] = (
                sat['position'][0] + np.random.uniform(-10, 10),
                sat['position'][1] + np.random.uniform(-10, 10),
                sat['position'][2]
            )

        # Battery drain over time
        for sat in self.satellites:
            sat['battery'] = max(0, sat['battery'] - 0.5)

        done = self.current_step >= self.max_steps or all(s['battery'] <= 0 for s in self.satellites)

        obs = self._get_observation()
        reward = Reward(value=reward_value, components=reward_components)
        info = {'step': self.current_step}

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            'satellites': self.satellites,
            'time_step': self.current_step,
            'ground_stations': self.ground_stations,
            'weather': self.weather,
            'pending_tasks': self.pending_tasks
        }

    def _get_observation(self) -> Observation:
        sat_states = [SatelliteState(**s) for s in self.satellites]
        return Observation(
            satellites=sat_states,
            time_step=self.current_step,
            ground_stations=self.ground_stations,
            weather_conditions=self.weather,
            pending_tasks=self.pending_tasks
        )

    def _can_downlink(self, sat_id: int) -> bool:
        sat = self.satellites[sat_id]
        # Simple check: if near a ground station
        for gs in self.ground_stations:
            dist = np.linalg.norm(np.array(sat['position'][:2]) - np.array(gs))
            if dist < 500:  # arbitrary distance
                return True
        return False