#!/usr/bin/env python3
"""
Baseline script using Groq HTTP completions (single clean implementation).

- Uses GROQ_API_KEY / GROQ_API_URL / MODEL_NAME (or GROQ_MODEL)
- Supports offline mode via GROQ_OFFLINE=1 to avoid network calls during dev
- Writes results to baseline_results.json
"""

import os
import json
import textwrap
from typing import Any, Dict

import requests
from requests.exceptions import RequestException
from dotenv import load_dotenv

from satellite_env import SatelliteConstellationEnv, Action, EasyTask, MediumTask, HardTask, TaskGrader

# Load environment variables from .env if present
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("GROQ_MODEL") or "groq-1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL") or os.getenv("GROQ_API_BASE")
GROQ_OFFLINE = os.getenv("GROQ_OFFLINE", "0") in ("1", "true", "True")

if not GROQ_API_KEY and not GROQ_OFFLINE:
    raise RuntimeError("GROQ_API_KEY is required unless GROQ_OFFLINE=1 is set")

if not GROQ_API_URL:
    GROQ_API_URL = f"https://api.groq.ai/v1/models/{MODEL_NAME}/completions"

HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}


def build_prompt(state: Dict[str, Any], desc: str) -> str:
    return textwrap.dedent(f"""
    Task: {desc}

    State:
    {json.dumps(state, indent=2)}

    Return a JSON object: {{"satellite_actions": {{"0": "capture", "1": "idle"}}}}
    """
    )


def parse_response_text(text: str, num_satellites: int) -> Action:
    try:
        parsed = json.loads(text)
    except Exception:
        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return Action(satellite_actions={i: "idle" for i in range(num_satellites)})
        parsed = json.loads(m.group(0))

    actions = {}
    for i in range(num_satellites):
        key = str(i)
        act = parsed.get("satellite_actions", {}).get(key, "idle")
        actions[i] = act if act in ("capture", "downlink", "maintain", "idle") else "idle"
    return Action(satellite_actions=actions)


def get_action(state: Dict[str, Any], desc: str) -> Action:
    num_sat = len(state.get("satellites", []))
    if GROQ_OFFLINE:
        return Action(satellite_actions={i: "idle" for i in range(num_sat)})

    payload = {"prompt": build_prompt(state, desc), "max_tokens": 256, "temperature": 0.1}
    try:
        r = requests.post(GROQ_API_URL, headers=HEADERS, json=payload, timeout=10)
        r.raise_for_status()
        body = r.json()

        text = None
        if isinstance(body, dict):
            if "choices" in body and body["choices"]:
                c = body["choices"][0]
                text = c.get("text") or c.get("message", {}).get("content")
            elif "output" in body:
                out = body["output"]
                text = out[0] if isinstance(out, list) and out else (out if isinstance(out, str) else None)

        if not text:
            text = json.dumps(body)

        return parse_response_text(text, num_sat)

    except RequestException as ex:
        print(f"Network/Groq request failed: {ex}")
        return Action(satellite_actions={i: "idle" for i in range(num_sat)})
    except Exception as ex:
        print(f"Unexpected parsing error: {ex}")
        return Action(satellite_actions={i: "idle" for i in range(num_sat)})


def run_task(task, name: str) -> float:
    env = SatelliteConstellationEnv()
    task.setup_environment(env)
    grader = TaskGrader(task)
    obs = env.reset()
    done = False
    actions_taken = []
    while not done:
        st = env.state()
        action = get_action(st, task.description)
        obs, reward, done, info = env.step(action)
        actions_taken.append(action)
    score = grader.grade_episode(env, actions_taken, env.state())
    print(f"{name} score: {score:.3f}")
    return score


def main() -> None:
    tasks = [(EasyTask(), "easy"), (MediumTask(), "medium"), (HardTask(), "hard")]
    results = {}
    for t, name in tasks:
        try:
            results[name] = run_task(t, name)
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = 0.0

    print("\nFinal Results:")
    for task_name, score in results.items():
        print(f"{task_name}: {score:.3f}")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()