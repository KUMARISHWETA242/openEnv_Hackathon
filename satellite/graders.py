"""Deterministic graders for satellite tasks."""

from .env import SatelliteTaskEnv
from .tasks import Task


class TaskGrader:
    """Grade a completed or partial episode on a 0.0-1.0 scale."""

    def __init__(self, task: Task):
        self.task = task

    def grade_episode(self, env: SatelliteTaskEnv) -> float:
        criteria = self.task.get_success_criteria()
        state = env.state()
        metrics = state.metrics

        scores = []

        if "min_images_captured" in criteria:
            scores.append(
                min(1.0, metrics.get("capture_task_completions", 0.0) / criteria["min_images_captured"])
            )

        if "min_data_downlinked" in criteria:
            scores.append(
                min(1.0, metrics.get("downlink_units", 0.0) / criteria["min_data_downlinked"])
            )

        if "min_tasks_completed" in criteria:
            scores.append(
                min(1.0, metrics.get("tasks_completed", 0.0) / criteria["min_tasks_completed"])
            )

        if "min_battery_final" in criteria:
            avg_battery = 0.0
            if state.satellites:
                avg_battery = sum(sat.battery for sat in state.satellites) / len(state.satellites)
            scores.append(min(1.0, avg_battery / criteria["min_battery_final"]))

        if "max_invalid_actions" in criteria:
            invalid_actions = metrics.get("invalid_actions", 0.0)
            allowance = max(1.0, float(criteria["max_invalid_actions"]))
            penalty_score = max(0.0, 1.0 - invalid_actions / allowance)
            scores.append(penalty_score)

        if not scores:
            return 0.0

        score = sum(scores) / len(scores)

        if state.step_count > criteria.get("max_steps", state.max_steps):
            score *= 0.8

        return max(0.0, min(1.0, score))
