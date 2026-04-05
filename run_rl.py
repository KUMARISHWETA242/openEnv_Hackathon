import os
import random
import time
import json
import re
from typing import List, Literal

from pydantic import BaseModel

# Import the environment package. It loads .env automatically via its _init_.
from satellite_env import SatelliteConstellationEnv, Action, EasyTask

SatelliteActionName = Literal['capture', 'downlink', 'maintain', 'idle']

# Try to import OpenAI client used in archive/check_hf_token.py pattern
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class SatelliteActions(BaseModel):
    satellite_actions: List[SatelliteActionName]


def _idle_actions(expected: int) -> List[SatelliteActionName]:
    return ["idle"] * expected


def _validate_satellite_actions(parsed, expected: int) -> List[SatelliteActionName]:
    if parsed is None:
        return _idle_actions(expected)

    try:
        if hasattr(SatelliteActions, "model_validate"):
            validated = SatelliteActions.model_validate(parsed)
        else:
            validated = SatelliteActions.parse_obj(parsed)
    except Exception:
        return _idle_actions(expected)

    actions = list(validated.satellite_actions)
    if len(actions) < expected:
        actions.extend(["idle"] * (expected - len(actions)))
    return actions[:expected]


def parse_satellite_actions(text, expected: int) -> List[SatelliteActionName]:
    if text is None:
        return _idle_actions(expected)

    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )

    if not isinstance(text, str):
        return _validate_satellite_actions(text, expected)

    text = text.strip()
    if not text:
        return _idle_actions(expected)

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None

    return _validate_satellite_actions(parsed, expected)


def build_hf_client():
    """Create a HuggingFace router-backed OpenAI client if HF_TOKEN is present.

    Returns the client or None if not available.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or OpenAI is None:
        return None

    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)


def build_observation_summary(observation) -> str:
    lines = []
    for sat in observation.satellites:
        lines.append(
            f"satellite {sat.id}: battery={sat.battery:.1f}, storage={sat.storage:.1f}, "
            f"last_action={sat.last_action}, position=({sat.position[0]:.1f}, {sat.position[1]:.1f}, {sat.position[2]:.1f})"
        )
    return "\n".join(lines)


def ask_llm_for_action(client, observation) -> List[SatelliteActionName]:
    """Query the LLM to get actions for satellites.

    The prompt is intentionally simple: list each satellite id and desired action.
    If the LLM call fails, raise the exception to let caller handle fallback.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict JSON-only assistant that controls satellites. "
                "Reply with exactly one JSON object and no extra text. "
                "The object must be of the form "
                "{\"satellite_actions\": [\"capture\", \"idle\"]}. "
                "Return satellite_actions as a list in satellite id order. "
                "Each item must be one of: capture, downlink, maintain, idle."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Time step: {observation.time_step}\n"
                f"Pending tasks: {json.dumps(observation.pending_tasks)}\n"
                f"Ground stations: {json.dumps(observation.ground_stations)}\n"
                f"Weather: {json.dumps(observation.weather_conditions)}\n"
                f"Satellite states:\n{build_observation_summary(observation)}\n"
                f"There are {len(observation.satellites)} satellites with ids {[s.id for s in observation.satellites]}. "
                "Return exactly one action per satellite in id order. "
                "Prefer capture when battery is comfortably above 10 and storage is below 90. "
                "Prefer maintain when battery is low. "
                "Prefer downlink when storage is high and a satellite may be in range of a ground station. "
                "Avoid returning all idle unless every satellite is resource-constrained. "
                "Example valid reply: "
                "{\"satellite_actions\": [\"capture\", \"maintain\", \"idle\"]}"
            ),
        }
    ]

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B:together")

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=120,
    )

    text = completion.choices[0].message.content

    return parse_satellite_actions(text, len(observation.satellites))


def heuristic_policy(observation) -> List[SatelliteActionName]:
    actions: List[SatelliteActionName] = []
    has_downlink_task = any(task.get("type") == "data_downlink" for task in observation.pending_tasks)
    has_capture_task = any(task.get("type") == "image_capture" for task in observation.pending_tasks)

    for sat in observation.satellites:
        if sat.battery <= 15:
            actions.append("maintain")
        elif sat.storage >= 80 and has_downlink_task:
            actions.append("downlink")
        elif has_capture_task and sat.battery > 20 and sat.storage < 80:
            actions.append("capture")
        elif sat.storage >= 50 and sat.battery > 10 and has_downlink_task:
            actions.append("downlink")
        elif sat.battery < 40:
            actions.append("maintain")
        elif sat.storage < 90:
            actions.append("capture")
        else:
            actions.append("idle")

    return actions


def should_override_idle(actions: List[SatelliteActionName], observation) -> bool:
    if any(action != "idle" for action in actions):
        return False

    for sat in observation.satellites:
        if sat.battery > 20 and sat.storage < 80:
            return True
        if sat.storage >= 50 and sat.battery > 10:
            return True

    return False


def run_episode(max_steps: int = 100):
    env = SatelliteConstellationEnv()
    # configure an easy task for demo
    task = EasyTask()
    task.setup_environment(env)

    client = build_hf_client()

    obs = env.reset()
    total_reward = 0.0

    for step in range(env.max_steps):
        # ask LLM for action if available
        action_map = None
        if client is not None:
            try:
                action_map = ask_llm_for_action(client, obs)
                if should_override_idle(action_map, obs):
                    action_map = heuristic_policy(obs)
            except Exception as e:
                print(f"LLM call failed: {e}; falling back to heuristic policy.")
                action_map = heuristic_policy(obs)
        else:
            action_map = heuristic_policy(obs)

        action = Action(satellite_actions=action_map)
        print(f"Actions: {dict(enumerate(action.satellite_actions))}")
        print("="*25)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        print(f"Step {step+1}/{env.max_steps} reward={reward.value:.2f} components={reward.components}")
        time.sleep(0.1)
        if done:
            print("Episode finished: done=True")
            break

    print(f"Total reward: {total_reward:.2f}")


if __name__ == '__main__':
    run_episode()
