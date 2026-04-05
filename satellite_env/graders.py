from typing import Dict, Any, List
from .env import SatelliteConstellationEnv, Action
from .tasks import Task

class TaskGrader:
    def __init__(self, task: Task):
        self.task = task

    def grade_episode(self, env: SatelliteConstellationEnv, actions: List[Action], final_state: Dict[str, Any]) -> float:
        """Grade the episode based on task criteria. Returns score 0.0-1.0"""
        criteria = self.task.get_success_criteria()

        score = 0.0
        max_score = len(criteria)

        # Count images captured (from actions)
        images_captured = sum(1 for action in actions for act in action.satellite_actions if act == 'capture')

        # Count data downlinked (from actions)
        data_downlinked = sum(1 for action in actions for act in action.satellite_actions if act == 'downlink')

        # Final battery levels
        final_batteries = [s['battery'] for s in final_state['satellites']]
        avg_final_battery = sum(final_batteries) / len(final_batteries) if final_batteries else 0

        # Check criteria
        if 'min_images_captured' in criteria:
            if images_captured >= criteria['min_images_captured']:
                score += 1.0
            else:
                score += images_captured / criteria['min_images_captured']

        if 'min_data_downlinked' in criteria:
            if data_downlinked >= criteria['min_data_downlinked']:
                score += 1.0
            else:
                score += data_downlinked / criteria['min_data_downlinked']

        if 'min_battery_final' in criteria:
            if avg_final_battery >= criteria['min_battery_final']:
                score += 1.0
            else:
                score += avg_final_battery / criteria['min_battery_final']

        # Penalty for exceeding max steps
        if final_state['time_step'] > criteria.get('max_steps', 100):
            score *= 0.8

        return min(1.0, score / max_score)
