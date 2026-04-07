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
import ast

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

    Return a JSON object: {{"satellite_actions": {{"0": "capture", "1": "idle"}}}}
    """
    )


def parse_response_text(text: str, num_satellites: int):
    # Accept either a parsed dict or a string to be parsed
    parsed = None
    if isinstance(text, dict):
        parsed = text
    else:
        # Try JSON
        try:
            parsed = json.loads(text)
        except Exception:
            # Try Python literal (handles single-quoted dicts)
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                # Try to extract a JSON-looking substring
                import re

                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    snippet = m.group(0)
                    try:
                        parsed = json.loads(snippet)
                    except Exception:
                        try:
                            parsed = ast.literal_eval(snippet)
                        except Exception:
                            parsed = None

    if not parsed or not isinstance(parsed, dict):
        # couldn't parse — indicate failure to caller by returning None
        return None

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
    # Build a compact prompt used for HF inference endpoints
    prompt = "\n".join([
        f"[system] Task: {desc}",
        f"[user] {json.dumps(state)}",
    ])

    # Use a strict system instruction with an example to encourage
    # a JSON-only response on the first call. Low temperature and a
    # concise max_tokens increase determinism.
    system_msg = (
        "You are a strict JSON-only assistant.\n"
        "When asked, return ONLY a single valid JSON object and nothing else.\n"
        "The JSON must have the shape: {\"satellite_actions\": {\"0\": \"capture\", \"1\": \"idle\"}}\n"
        "Always use double quotes for keys and string values.\n"
        "If you cannot determine an action, use \"idle\" for that satellite.\n"
        "Example input -> output:\n"
        "Input State: {'satellites': [{'id':0},{'id':1}], 'other': '...'}\n"
        "Output: {\"satellite_actions\": {\"0\": \"idle\", \"1\": \"idle\"}}\n"
    )

    user_msg = f"State: {json.dumps(state)}\n\nReturn a JSON object with key 'satellite_actions' mapping satellite indices (as strings) to actions (capture/downlink/maintain/idle). Reply ONLY with the JSON object and no surrounding text."

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            max_tokens=200,
            temperature=0.0,
            timeout=3,
        )

        # If the wrapper returned a dict with 'choices' or model-specific keys,
        # try to extract a textual response or pass the dict to the parser.
        if isinstance(resp, dict):
            # If it's an HF inference-style dict with generated_text, use it
            if "choices" in resp and len(resp["choices"]) > 0:
                first = resp["choices"][0]
                msg = first.get("message") or {}
                response_text = msg.get("content") or msg.get("text") or first.get("text")
                if response_text:
                    return parse_response_text(response_text, num_sat)
                # else fallthrough to parsing the dict
                return parse_response_text(resp, num_sat)

            # HF inference model responses sometimes return [{'generated_text': '...'}]
            if isinstance(resp, list):
                try:
                    return parse_response_text(resp[0], num_sat)
                except Exception:
                    return parse_response_text(json.dumps(resp), num_sat)

            # Otherwise attempt to parse the dict directly
            return parse_response_text(resp, num_sat)

        # object-like response (OpenAI SDK style)
        text = None
        if resp and getattr(resp, "choices", None):
            first = resp.choices[0]
            msg = getattr(first, "message", None) or first
            text = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)) or first.get("text", "")

        if text:
            parsed = parse_response_text(text, num_sat)
            if parsed:
                return parsed

        # last resort: try to parse dict/list responses
        parsed = parse_response_text(resp, num_sat)
        if parsed:
            return parsed

        # fallback: deterministic local policy — choose satellites with highest battery to 'capture'
        sats = state.get("satellites", [])
        battery_pairs = [(i, s.get("battery", 0)) for i, s in enumerate(sats)]
        battery_pairs.sort(key=lambda x: x[1], reverse=True)
        num_to_capture = max(1, len(battery_pairs) // 3)
        actions = {i: "idle" for i in range(num_sat)}
        for i, _ in battery_pairs[:num_to_capture]:
            actions[i] = "capture"
        return Action(satellite_actions=actions)

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