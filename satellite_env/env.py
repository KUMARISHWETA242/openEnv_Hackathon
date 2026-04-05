from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
import numpy as np

class SatelliteState(BaseModel):
    id: int
    position: Tuple[float, float, float]  # latitude, longitude, altitude_km
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
        self.earth_radius_km = 6371.0
        self.downlink_range_km = 2200.0
        self.num_satellites = num_satellites
        self.max_steps = max_steps
        self.current_step = 0
        self.satellites = []
        self.ground_stations = [
            (28.6139, 77.2090),   # New Delhi
            (1.3521, 103.8198),   # Singapore
            (34.0522, -118.2437), # Los Angeles
        ]
        self.weather = {"region1": 0.2, "region2": 0.5}  # cloud cover
        self.pending_tasks = []
        self._reset_satellites()

    def _reset_satellites(self):
        self.satellites = []
        for i in range(self.num_satellites):
            altitude = np.random.uniform(450, 650)
            inclination = np.random.uniform(-65, 65)
            phase = np.random.uniform(0, 2 * np.pi)
            angular_velocity = np.random.uniform(6, 14)
            position = self._orbit_position_from_phase(phase, inclination, altitude)
            self.satellites.append({
                'id': i,
                'position': position,
                'battery': 100.0,
                'storage': 0.0,
                'last_action': 'idle',
                'orbit_phase': phase,
                'angular_velocity': angular_velocity,
                'inclination': inclination,
                'altitude': altitude,
            })

    def _orbit_position_from_phase(self, phase: float, inclination: float, altitude: float) -> Tuple[float, float, float]:
        latitude = float(np.clip(inclination * np.sin(phase), -85.0, 85.0))
        longitude = float(((np.degrees(phase) % 360.0) + 180.0) % 360.0 - 180.0)
        return (latitude, longitude, altitude)

    def _advance_satellite_orbit(self, sat: Dict[str, Any]) -> None:
        sat['orbit_phase'] = (sat['orbit_phase'] + np.deg2rad(sat['angular_velocity'])) % (2 * np.pi)
        sat['position'] = self._orbit_position_from_phase(
            sat['orbit_phase'],
            sat['inclination'],
            sat['altitude'],
        )

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

        # Update positions using a lightweight orbital ground-track approximation
        for sat in self.satellites:
            self._advance_satellite_orbit(sat)

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
        sat_lat, sat_lon, _ = sat['position']
        # Simple check: if near a ground station
        for gs in self.ground_stations:
            dist = self._ground_distance_km(sat_lat, sat_lon, gs[0], gs[1])
            if dist < self.downlink_range_km:
                return True
        return False

    def _ground_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        hav = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        central_angle = 2 * np.arctan2(np.sqrt(hav), np.sqrt(1 - hav))
        return float(self.earth_radius_km * central_angle)
