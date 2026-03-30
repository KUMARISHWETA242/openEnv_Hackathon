#!/usr/bin/env python3
"""
Inference Script for Satellite Constellation Management Environment
===================================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import re
import textwrap
from typing import List, Dict, Any

import json
import requests
from dotenv import load_dotenv

from satellite_env import SatelliteConstellationEnv, Action, EasyTask, MediumTask, HardTask

load_dotenv()

# Groq configuration
GROQ_API_URL = os.getenv("GROQ_API_URL") or os.getenv("API_BASE_URL") or os.getenv("GROQ_API_BASE")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("GROQ_MODEL") or "groq-1"
MAX_STEPS = 50
TEMPERATURE = 0.2
MAX_TOKENS = 300
FALLBACK_ACTION = {"satellite_actions": {}}

DEBUG = True

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are managing a satellite constellation with limited resources.
    Your goal is to maximize mission value by making optimal decisions for each satellite.

    Available actions per satellite: 'capture', 'downlink', 'maintain', 'idle'
    - 'capture': Take an image (costs battery, fills storage)
    - 'downlink': Send data to ground station (if in range, costs battery, frees storage)
    - 'maintain': Perform maintenance (recharges battery)
    - 'idle': No action

    Reply with a JSON object containing 'satellite_actions' as a dict of satellite_id -> action.
    Example: {"satellite_actions": {"0": "capture", "1": "downlink", "2": "maintain"}}

    Consider battery levels, storage usage, and communication opportunities.
    Do not include explanations or additional text.
    """
).strip()


def build_satellite_status(observation) -> str:
    """Build a readable status of all satellites."""
    status_lines = []
    for sat in observation.satellites:
        status_lines.append(
            f"  Satellite {sat.id}: pos=({sat.position[0]:.1f}, {sat.position[1]:.1f}, {sat.position[2]:.1f}), "
            f"battery={sat.battery:.1f}%, storage={sat.storage:.1f}%, last_action={sat.last_action}"
        )
    return "\n".join(status_lines)


def build_ground_stations_status(observation) -> str:
    """Build status of ground stations."""
    if not observation.ground_stations:
        return "No ground stations available"
    stations = [f"({lat:.1f}, {lon:.1f})" for lat, lon in observation.ground_stations]
    return f"Ground stations: {', '.join(stations)}"


def build_weather_status(observation) -> str:
    """Build weather conditions status."""
    if not observation.weather_conditions:
        return "Weather: Clear"
    conditions = [f"{region}: {cover:.2f}" for region, cover in observation.weather_conditions.items()]
    return f"Weather conditions: {', '.join(conditions)}"


def build_tasks_status(observation) -> str:
    """Build pending tasks status."""
    if not observation.pending_tasks:
        return "No pending tasks"
    tasks = [f"{task.get('type', 'unknown')} in {task.get('region', 'unknown')}" for task in observation.pending_tasks]
    return f"Pending tasks: {', '.join(tasks)}"


def build_user_prompt(step: int, observation, history: List[str], task_description: str) -> str:
    """Build the user prompt for the model."""

    satellite_status = build_satellite_status(observation)
    ground_stations = build_ground_stations_status(observation)
    weather = build_weather_status(observation)
    tasks = build_tasks_status(observation)

    history_text = "\n".join(history[-3:]) if history else "None"

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Mission Goal: {task_description}

        Current Status:
        {satellite_status}
        {ground_stations}
        {weather}
        {tasks}

        Previous actions:
        {history_text}

        Reply with a JSON object containing satellite_actions as a dict of satellite_id -> action.
        Available actions: 'capture', 'downlink', 'maintain', 'idle'
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str, num_satellites: int) -> Dict[str, Any]:
    """Parse the model's response into an Action object."""
    if not response_text:
        return FALLBACK_ACTION

    try:
        # Try to parse as JSON
        result = json.loads(response_text.strip())
        if "satellite_actions" in result:
            # Validate that we have actions for satellites that exist
            actions = {}
            for sat_id in range(num_satellites):
                sat_id_str = str(sat_id)
                if sat_id_str in result["satellite_actions"]:
                    action = result["satellite_actions"][sat_id_str]
                    if action in ['capture', 'downlink', 'maintain', 'idle']:
                        actions[sat_id] = action
                    else:
                        actions[sat_id] = 'idle'  # fallback
                else:
                    actions[sat_id] = 'idle'  # default
            return {"satellite_actions": actions}
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON from text
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if "satellite_actions" in result:
                return result
        except json.JSONDecodeError:
            pass

    # Final fallback
    return FALLBACK_ACTION


def run_inference(task_name: str = "easy") -> Dict[str, Any]:
    """Run inference on the specified task."""

    # Initialize Groq client (HTTP)
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY or API_KEY environment variable not set")

    if not GROQ_API_URL:
        GROQ_API_URL = f"https://api.groq.ai/v1/models/{MODEL_NAME}/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Set up task
    if task_name == "easy":
        task = EasyTask()
    elif task_name == "medium":
        task = MediumTask()
    elif task_name == "hard":
        task = HardTask()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Initialize environment
    env = SatelliteConstellationEnv()
    task.setup_environment(env)

    history: List[str] = []
    total_reward = 0.0

    try:
        # Reset environment
        observation = env.reset()
        print(f"Starting {task_name} task: {task.description}")

        for step in range(1, MAX_STEPS + 1):
            # Build prompt
            user_prompt = build_user_prompt(step, observation, history, task.description)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            try:
                payload = {
                    "prompt": SYSTEM_PROMPT + "\n" + user_prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                }
                resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                body = resp.json()
                # extract text
                response_text = ""
                if isinstance(body, dict):
                    if "choices" in body and len(body["choices"]) > 0:
                        first = body["choices"][0]
                        response_text = first.get("text") or first.get("message", {}).get("content", "")
                    elif "output" in body:
                        out = body["output"]
                        response_text = out[0] if isinstance(out, list) and len(out) > 0 else (out if isinstance(out, str) else "")
                if not response_text:
                    response_text = json.dumps(body)
            except Exception as exc:
                print(f"Model request failed ({exc}). Using fallback action.")
                response_text = json.dumps(FALLBACK_ACTION)

            # Parse action
            action_dict = parse_model_action(response_text, env.num_satellites)
            action = Action(**action_dict)

            if DEBUG:
                print(f"Step {step}: {action_dict}")

            # Step environment
            observation, reward, done, info = env.step(action)

            total_reward += reward.value

            # Record history
            history_line = f"Step {step}: {action_dict} -> reward {reward.value:.2f}"
            history.append(history_line)

            if done:
                print(f"Episode complete at step {step}. Total reward: {total_reward:.2f}")
                break

        else:
            print(f"Reached max steps ({MAX_STEPS}). Total reward: {total_reward:.2f}")

    except Exception as e:
        print(f"Error during inference: {e}")
        return {"error": str(e), "score": 0.0}

    # Return results
    final_state = env.state()
    return {
        "task": task_name,
        "total_reward": total_reward,
        "steps": len(history),
        "final_battery_avg": sum(s['battery'] for s in final_state['satellites']) / len(final_state['satellites']),
        "history": history
    }


def main() -> None:
    """Main function to run inference on all tasks."""

    if not all([GROQ_API_URL, GROQ_API_KEY, MODEL_NAME]):
        print("Error: Missing required environment variables:")
        print("  GROQ_API_URL (or API_BASE_URL), GROQ_API_KEY (or API_KEY/HF_TOKEN), MODEL_NAME")
        return

    tasks = ["easy", "medium", "hard"]
    results = {}

    for task_name in tasks:
        print(f"\n{'='*50}")
        print(f"Running {task_name} task...")
        print(f"{'='*50}")

        result = run_inference(task_name)
        results[task_name] = result

        if "error" not in result:
            print(f"Completed {task_name}: {result['total_reward']:.2f} reward, {result['steps']} steps")
        else:
            print(f"Failed {task_name}: {result['error']}")

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("INFERENCE COMPLETE")
    print("Results saved to inference_results.json")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()