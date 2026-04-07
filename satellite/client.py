"""WebSocket client for the satellite OpenEnv server."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SatelliteAction, SatelliteObservation


class SatelliteEnv(EnvClient[SatelliteAction, SatelliteObservation, State]):
    """Client for interacting with a running satellite environment server."""

    def _step_payload(self, action: SatelliteAction) -> Dict[str, Any]:
        """Convert a typed action to the server step payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SatelliteObservation]:
        """Parse a server response into a typed step result."""
        observation_data = payload.get("observation", payload)
        observation = SatelliteObservation.model_validate(
            {
                **observation_data,
                "done": payload.get("done", observation_data.get("done", False)),
                "reward": payload.get("reward", observation_data.get("reward", 0.0)),
                "metadata": observation_data.get("metadata", {}),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse state payload into the OpenEnv state type."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
