#!/usr/bin/env python3
"""
Baseline script using Hugging Face Inference HTTP completions (single clean implementation).

- Uses HF_TOKEN / API_BASE_URL / MODEL_NAME (or HF_MODEL)
- Supports offline mode via GROQ_OFFLINE=1 to avoid network calls during dev
- Writes results to baseline_results.json
"""

import os
import json
import textwrap
from typing import Any, Dict

import requests
from openai import OpenAI
from requests.exceptions import RequestException
from dotenv import load_dotenv

from satellite_env import SatelliteConstellationEnv, Action, EasyTask, MediumTask, HardTask, TaskGrader

# Load environment variables from .env if present
load_dotenv()

# Hugging Face config
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("HF_MODEL") or "gpt2"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL")
GROQ_OFFLINE = os.getenv("GROQ_OFFLINE", "0") in ("1", "true", "True")

if not os.getenv("HF_TOKEN") and not GROQ_OFFLINE:
    raise RuntimeError("HF_TOKEN (or OPENAI_API_KEY/API_KEY) is required unless GROQ_OFFLINE=1 is set")

# instantiate the provided wrapper client pointing to HF router
try:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )
except KeyError:
    if not GROQ_OFFLINE:
        raise ValueError("HF_TOKEN environment variable not set; required to instantiate OpenAI wrapper client")


def build_prompt(state: Dict[str, Any], desc: str) -> str:
    return textwrap.dedent(f"""
    Task: {desc}

    State:
    {json.dumps(state, indent=2)}

    Return only a valid JSON object with the following format, no additional text or explanation:
    {{"satellite_actions": {{"0": "capture", "1": "idle"}}}}
    """
    )


def parse_response_text(text: str, num_satellites: int) -> Action:
    print(f"Raw response text: {repr(text)}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            print("No JSON found in response")
            return Action(satellite_actions={i: "idle" for i in range(num_satellites)})
        json_str = m.group(0)
        print(f"Extracted JSON: {repr(json_str)}")
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e2:
            print(f"Failed to parse extracted JSON: {e2}")
            return Action(satellite_actions={i: "idle" for i in range(num_satellites)})

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

    try:
        prompt = build_prompt(state, desc)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )

        text = None
        if resp and getattr(resp, "choices", None):
            first = resp.choices[0]
            msg = getattr(first, "message", None) or first
            text = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)) or first.get("text", "")

        if not text:
            text = json.dumps(resp)

        return parse_response_text(text, num_sat)

    except RequestException as ex:
        print(f"Network request failed: {ex}")
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
        print(f"Reward: {reward}")
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