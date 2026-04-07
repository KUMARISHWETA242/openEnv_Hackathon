"""Baseline inference runner for all satellite tasks."""

import json
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from satellite import (
    EasyTask,
    HardTask,
    MediumTask,
    SatelliteAction,
    SatelliteTaskEnv,
    TaskGrader,
)

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")
BASELINE_POLICY = os.getenv("BASELINE_POLICY", "openai").lower()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "350"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.0"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
FALLBACK_ACTION = "idle"
TASK_ORDER = ["easy", "medium", "hard"]
TASK_TYPES = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}

ACTION_PATTERN = re.compile(r"(capture|downlink|maintain|idle)", re.IGNORECASE)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are managing a real-world satellite constellation.
    Your objective is to complete visible tasks while preserving battery and storage health.

    Available actions per satellite:
    - capture
    - downlink
    - maintain
    - idle

    Rules of thumb:
    - prioritize capture when image tasks remain and the satellite has battery and storage headroom
    - prioritize downlink when storage is high or downlink tasks remain
    - maintain when battery is low
    - avoid repeated wasteful actions and invalid moves

    Respond with JSON only in the form {"0": "capture", "1": "idle"}.
    """
).strip()


@dataclass
class TaskRunResult:
    task_name: str
    score: float
    total_reward: float
    steps: int
    done: bool
    metrics: Dict[str, float]


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-6:])


def format_observation(task_name: str, observation: Dict[str, Any]) -> str:
    satellites_info = []
    for sat in observation.get("satellites", []):
        satellites_info.append(
            f"  Satellite {sat['id']}: battery={sat['battery']:.1f}, storage={sat['storage']:.1f}, last={sat['last_action']}"
        )

    tasks_info = []
    for task in observation.get("pending_tasks", [])[:6]:
        descriptor = task["type"]
        if task["type"] == "image_capture":
            descriptor += f" region={task.get('region')}"
        if task["type"] == "data_downlink":
            descriptor += f" station={task.get('station')}"
            descriptor += f" units={task.get('units_remaining', 0)}"
        tasks_info.append(f"  - {descriptor} priority={task.get('priority', 1)}")

    weather_info = ", ".join(
        f"{region}={cover:.0%}"
        for region, cover in observation.get("weather_conditions", {}).items()
    )

    return textwrap.dedent(
        f"""
        Task: {task_name}
        Time Step: {observation.get('time_step', 0)}
        Total Reward: {observation.get('total_reward', 0.0):.2f}
        Weather: {weather_info or 'n/a'}

        Satellites:
        {chr(10).join(satellites_info)}

        Pending Tasks:
        {chr(10).join(tasks_info) or '  - None'}
        """
    ).strip()


def build_user_prompt(
    task_name: str,
    step: int,
    observation: Dict[str, Any],
    history: List[str],
    total_reward: float,
) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Aggregate reward so far: {total_reward:+.2f}

        Current state:
        {format_observation(task_name, observation)}

        Recent action history:
        {build_history_lines(history)}

        Return JSON only.
        """
    ).strip()


def heuristic_action(observation: Dict[str, Any]) -> Dict[int, str]:
    actions: Dict[int, str] = {}
    has_capture_task = any(task["type"] == "image_capture" for task in observation["pending_tasks"])
    has_downlink_task = any(task["type"] == "data_downlink" for task in observation["pending_tasks"])

    for sat in observation["satellites"]:
        sat_id = sat["id"]
        battery = sat["battery"]
        storage = sat["storage"]
        if battery < 20:
            actions[sat_id] = "maintain"
        elif storage > 60 and has_downlink_task:
            actions[sat_id] = "downlink"
        elif has_capture_task and battery > 25 and storage < 75:
            actions[sat_id] = "capture"
        elif has_downlink_task and storage > 0:
            actions[sat_id] = "downlink"
        elif battery < 45:
            actions[sat_id] = "maintain"
        else:
            actions[sat_id] = "idle"
    return actions


def parse_model_action(response_text: str, observation: Dict[str, Any]) -> Dict[int, str]:
    if not response_text:
        return heuristic_action(observation)

    try:
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            actions: Dict[int, str] = {}
            for key, value in parsed.items():
                action = str(value).strip().lower()
                if action not in {"capture", "downlink", "maintain", "idle"}:
                    action = FALLBACK_ACTION
                actions[int(key)] = action
            if actions:
                return actions
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    extracted: Dict[int, str] = {}
    for line in response_text.splitlines():
        match = ACTION_PATTERN.search(line)
        if match:
            extracted[len(extracted)] = match.group(1).lower()
    return extracted or heuristic_action(observation)


def observation_to_dict(observation) -> Dict[str, Any]:
    return {
        "satellites": [
            {
                "id": sat.id,
                "position": sat.position,
                "battery": sat.battery,
                "storage": sat.storage,
                "last_action": sat.last_action,
            }
            for sat in observation.satellites
        ],
        "time_step": observation.time_step,
        "ground_stations": observation.ground_stations,
        "weather_conditions": observation.weather_conditions,
        "pending_tasks": observation.pending_tasks,
        "total_reward": observation.total_reward,
    }


def build_client() -> OpenAI | None:
    if BASELINE_POLICY == "heuristic":
        return None
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required for inference.")
    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME is required for inference.")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def choose_actions(
    client: OpenAI | None,
    task_name: str,
    step: int,
    observation: Dict[str, Any],
    history: List[str],
    total_reward: float,
) -> Dict[int, str]:
    prompt = build_user_prompt(task_name, step, observation, history, total_reward)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if client is None:
        return heuristic_action(observation)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    response_text = completion.choices[0].message.content or ""
    if DEBUG:
        print(f"[{task_name}] model response: {response_text[:200]}")
    return parse_model_action(response_text, observation)


def run_task(task_name: str, client: OpenAI | None) -> TaskRunResult:
    env = SatelliteTaskEnv(task_name=task_name)
    task = TASK_TYPES[task_name]()
    grader = TaskGrader(task)
    observation = env.reset()
    total_reward = 0.0
    history: List[str] = []
    step_limit = env.state().max_steps

    for step in range(1, step_limit + 1):
        obs_dict = observation_to_dict(observation)
        try:
            actions = choose_actions(client, task_name, step, obs_dict, history, total_reward)
        except Exception as exc:  # pragma: no cover - API failure fallback
            print(f"[{task_name}] request failed ({exc}); using deterministic heuristic fallback.")
            actions = heuristic_action(obs_dict)

        action = SatelliteAction(satellite_actions=actions)
        observation, reward, done, _ = env.step(action)
        total_reward += reward.value
        history.append(f"step {step}: {actions} -> {reward.value:+.2f}")

        if REQUEST_DELAY > 0 and step < step_limit and not done:
            time.sleep(REQUEST_DELAY)
        if done:
            break

    state = env.state()
    score = grader.grade_episode(env)
    return TaskRunResult(
        task_name=task_name,
        score=score,
        total_reward=state.total_reward,
        steps=state.step_count,
        done=state.done,
        metrics=state.metrics,
    )


def print_summary(results: List[TaskRunResult]) -> None:
    aggregate = sum(result.score for result in results) / len(results)
    print("\nBaseline Summary")
    print("=" * 50)
    for result in results:
        print(
            f"{result.task_name:<6} score={result.score:.4f} "
            f"reward={result.total_reward:.2f} steps={result.steps} done={result.done}"
        )
    print("-" * 50)
    print(f"aggregate_score={aggregate:.4f}")
    print("=" * 50)


def main() -> None:
    client = build_client()
    results = [run_task(task_name, client) for task_name in TASK_ORDER]
    print_summary(results)


if __name__ == "__main__":
    main()
