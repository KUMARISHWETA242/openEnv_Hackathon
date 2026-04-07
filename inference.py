"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import asyncio
import json
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from satellite import EasyTask, HardTask, MediumTask, SatelliteAction, TaskGrader
from satellite.client import SatelliteEnv
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_IMAGE = os.getenv("ENV_IMAGE", "satellite-openenv:latest")
BASELINE_POLICY = os.getenv("BASELINE_POLICY", "openai").lower()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
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
ACTION_PREFIX_RE = re.compile(r"^(action|next action)\s*[:\-]\s*", re.IGNORECASE)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are managing a real-world satellite constellation.
    Reply with exactly one JSON object mapping satellite ids to actions.

    Valid actions:
    - capture
    - downlink
    - maintain
    - idle

    Rules:
    - prioritize image capture when image tasks remain and the satellite has enough battery/storage
    - prioritize downlink when storage is high or data-downlink tasks remain
    - use maintain when battery is low
    - avoid repeated wasteful actions, invalid moves, and risky resource depletion

    Output format:
    {"0": "capture", "1": "idle"}

    Do not include explanations or any extra text outside the JSON object.
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
            f"  Satellite {sat['id']}: battery={sat['battery']:.1f}, "
            f"storage={sat['storage']:.1f}, last={sat['last_action']}"
        )

    tasks_info = []
    for task in observation.get("pending_tasks", [])[:8]:
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
        {chr(10).join(satellites_info) or '  None'}

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

        Previous steps:
        {build_history_lines(history)}

        Reply with exactly one JSON object.
        """
    ).strip()


def heuristic_action(observation: Dict[str, Any]) -> Dict[int, str]:
    actions: Dict[int, str] = {}
    has_capture_task = any(
        task["type"] == "image_capture" for task in observation["pending_tasks"]
    )
    has_downlink_task = any(
        task["type"] == "data_downlink" for task in observation["pending_tasks"]
    )

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

    cleaned = ACTION_PREFIX_RE.sub("", response_text.strip())

    try:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            actions: Dict[int, str] = {}
            for key, value in parsed.items():
                action = str(value).strip().lower()
                if not ACTION_PATTERN.fullmatch(action):
                    action = FALLBACK_ACTION
                actions[int(key)] = action
            if actions:
                return actions
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    extracted: Dict[int, str] = {}
    for line in cleaned.splitlines():
        match = ACTION_PATTERN.search(line)
        if match:
            extracted[len(extracted)] = match.group(1).lower()
    return extracted or heuristic_action(observation)


def observation_to_dict(observation: Any) -> Dict[str, Any]:
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


def build_client() -> Optional[OpenAI]:
    if BASELINE_POLICY == "heuristic":
        return None

    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not API_KEY:
        missing.append("HF_TOKEN")

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}."
        )

    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def choose_actions(
    client: Optional[OpenAI],
    task_name: str,
    step: int,
    observation: Dict[str, Any],
    history: List[str],
    total_reward: float,
) -> Dict[int, str]:
    if client is None:
        return heuristic_action(observation)

    user_prompt = build_user_prompt(task_name, step, observation, history, total_reward)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    response_text = completion.choices[0].message.content or ""
    if DEBUG:
        print(f"[{task_name}] model response: {response_text[:300]}")
    return parse_model_action(response_text, observation)


async def run_task(task_name: str, client: Optional[OpenAI]) -> TaskRunResult:
    env = await SatelliteEnv.from_docker_image(
        image=ENV_IMAGE,
        env_vars={"OPENENV_TASK_NAME": task_name},
    )
    task = TASK_TYPES[task_name]()
    grader = TaskGrader(task)
    history: List[str] = []
    total_reward = 0.0

    try:
        reset_result = await env.reset()
        observation = reset_result.observation
        state = await env.state()
        step_limit = getattr(state, "max_steps", None) or {"easy": 50, "medium": 100, "hard": 200}[task_name]

        print(f"\nRunning task: {task_name} on image {ENV_IMAGE}")

        for step in range(1, step_limit + 1):
            obs_dict = observation_to_dict(observation)
            try:
                actions = choose_actions(
                    client, task_name, step, obs_dict, history, total_reward
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{task_name}] Model request failed ({exc}). Using heuristic fallback.")
                actions = heuristic_action(obs_dict)

            print(f"Step {step}: model suggested -> {actions}")

            result = await env.step(SatelliteAction(satellite_actions=actions))
            observation = result.observation
            reward = result.reward or 0.0
            total_reward += reward
            history.append(f"step {step}: {actions} -> reward {reward:+.2f}")

            print(
                f"  Reward: {reward:+.2f} | Total: {total_reward:+.2f} "
                f"| Done: {result.done}"
            )

            if REQUEST_DELAY > 0 and step < step_limit and not result.done:
                time.sleep(REQUEST_DELAY)

            if result.done:
                print("Episode complete.")
                break
        else:
            print(f"Reached max steps ({step_limit}).")

        final_state = await env.state()
        metrics = {}
        if hasattr(observation, "metadata"):
            metrics = {
                key: float(value)
                for key, value in (observation.metadata.get("metrics", {}) or {}).items()
            }

        # Reconstruct a lightweight canonical env state for deterministic grading.
        # The grader only needs step count, metrics, and satellite battery values.
        class _Sat:
            def __init__(self, battery: float):
                self.battery = battery

        class _State:
            def __init__(self) -> None:
                self.metrics = metrics
                self.satellites = [_Sat(sat.battery) for sat in observation.satellites]
                self.step_count = getattr(final_state, "step_count", 0)
                self.max_steps = step_limit

        class _Env:
            def state(self) -> _State:
                return _State()

        score = grader.grade_episode(_Env())  # type: ignore[arg-type]
        return TaskRunResult(
            task_name=task_name,
            score=score,
            total_reward=observation.total_reward,
            steps=getattr(final_state, "step_count", 0),
            done=bool(getattr(observation, "done", False)),
            metrics=metrics,
        )
    finally:
        await env.close()


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


async def async_main() -> None:
    client = build_client()
    results = []
    for task_name in TASK_ORDER:
        results.append(await run_task(task_name, client))
    print_summary(results)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()