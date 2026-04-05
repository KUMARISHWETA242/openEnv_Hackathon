"""
Inference Script Example for Satellite Environment
===================================================
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
import time
import textwrap
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from satellite.server.satellite_environment import SatelliteEnvironment
from satellite.models import SatelliteAction

# Load environment variables from .env file
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 50
TEMPERATURE = 0.2
MAX_TOKENS = 300
FALLBACK_ACTION = "idle"

# Rate limiting - Add delay between API requests (in seconds)
# Increase this value to slow down requests and reduce API usage
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2.0"))  # Default 2 seconds between requests

# FREE ALTERNATIVE MODELS (if you run out of credits):
# 
# Option 1: Ollama (FREE - runs locally on your machine)
#   https://ollama.ai
#   Command: ollama pull mistral
#   Setup:
#     API_BASE_URL="http://localhost:11434/v1"
#     MODEL_NAME="mistral"
#   Start server: ollama serve
#
# Option 2: Groq (FREE tier - very fast, 25 requests/min limit)
#   https://console.groq.com
#   Setup:
#     API_BASE_URL="https://api.groq.com/openai/v1"
#     HF_TOKEN=<your_groq_api_key>
#     MODEL_NAME="mixtral-8x7b-32768"
#
# Option 3: Together AI (FREE tier)
#   https://api.together.xyz
#   Setup similar to above
#
# To use: Update .env file with the appropriate values

DEBUG = True
ACTION_PATTERN = re.compile(r"(capture|downlink|maintain|idle)", re.IGNORECASE)


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an intelligent satellite constellation manager.
    You control a constellation of satellites with the following capabilities:
    - capture: Take images (uses battery, increases storage)
    - downlink: Send data to ground station (uses battery, decreases storage)
    - maintain: Recharge battery (restores battery level)
    - idle: Do nothing (passive energy recovery)
    
    For each satellite, decide which action to take based on:
    - Battery level (critical: < 20%, warning: < 50%)
    - Storage level (full: > 90%, empty: < 10%)
    - Task priorities and weather conditions
    - Ground station availability
    
    Respond with a JSON dict mapping satellite_id to action.
    Example: {"0": "capture", "1": "downlink", "2": "maintain"}
    """
).strip()


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-6:])


def format_observation(observation: Dict[str, Any]) -> str:
    """Format observation for LLM context."""
    satellites_info = []
    for sat in observation.get("satellites", []):
        sat_str = (
            f"  Satellite {sat['id']}: "
            f"Battery={sat['battery']:.1f}%, "
            f"Storage={sat['storage']:.1f}%, "
            f"LastAction={sat['last_action']}"
        )
        satellites_info.append(sat_str)

    weather_info = ", ".join(
        f"{region}: {cover:.0%} cloud cover"
        for region, cover in observation.get("weather_conditions", {}).items()
    )

    tasks_info = "\n  ".join(
        f"{task['type']} (priority: {task.get('priority', 'unknown')})"
        for task in observation.get("pending_tasks", [])[:3]
    )

    return f"""
Satellites:
{chr(10).join(satellites_info)}

Weather: {weather_info}

Pending Tasks (top 3):
  {tasks_info or 'None'}

Time Step: {observation.get('time_step', 0)} / 100
Total Reward: {observation.get('total_reward', 0):.1f}
"""


def build_user_prompt(
    step: int, observation: Dict[str, Any], history: List[str], reward: float
) -> str:
    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Last Reward: {reward:+.2f}
        
        Current State:
        {format_observation(observation)}
        
        Action History (last 6 steps):
        {build_history_lines(history)}
        
        Decide actions for all satellites. Respond with JSON dict only.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> Dict[int, str]:
    """Parse LLM response to extract satellite actions."""
    satellite_actions = {}

    if not response_text:
        satellite_actions[0] = FALLBACK_ACTION
        return satellite_actions

    # Try to parse JSON dict
    try:
        import json

        # Extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            # Convert string keys to int and validate actions
            valid_actions = {"capture", "downlink", "maintain", "idle"}
            for key, action in parsed.items():
                sat_id = int(key) if isinstance(key, str) else key
                action_str = str(action).lower().strip()
                if action_str in valid_actions:
                    satellite_actions[sat_id] = action_str
                else:
                    satellite_actions[sat_id] = FALLBACK_ACTION

            if satellite_actions:
                return satellite_actions
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract individual actions
    lines = response_text.splitlines()
    for line in lines:
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(1).lower()
            satellite_actions[len(satellite_actions)] = action

    # If no actions found, default to idle
    if not satellite_actions:
        satellite_actions[0] = FALLBACK_ACTION

    return satellite_actions


def main() -> None:
    """Run satellite environment inference with LLM agent."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = SatelliteEnvironment(num_satellites=5, max_steps=100)

    history: List[str] = []
    total_episode_reward = 0.0

    try:
        observation = env.reset()
        obs_dict = {
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

        print(f"Episode started with {len(observation.satellites)} satellites")
        print(f"Goal: Maximize reward through optimal satellite management\n")

        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                print("Environment signalled done. Stopping early.")
                break

            user_prompt = build_user_prompt(step, obs_dict, history, total_episode_reward)

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                if DEBUG:
                    print(f"[Model Response] {response_text[:100]}...")
                # Add delay between requests to reduce API usage
                if step < MAX_STEPS:  # Don't delay after last step
                    time.sleep(REQUEST_DELAY)
            except Exception as exc:  # pylint: disable=broad-except
                failure_msg = f"Model request failed ({exc}). Using fallback action."
                print(failure_msg)
                print("\n💡 To fix this issue:")
                print("   1. INCREASE REQUEST_DELAY in .env file:")
                print("      REQUEST_DELAY=5  # Wait 5 seconds between requests")
                print("   2. OR use a FREE model alternative:")
                print("      a) Ollama (local): ollama pull mistral")
                print("         Then set: API_BASE_URL=http://localhost:11434/v1")
                print("      b) Groq (free/fast): Get API key at https://console.groq.com")
                print("         Then set: API_BASE_URL=https://api.groq.com/openai/v1")
                print("      c) Together AI: Get API key at https://api.together.xyz")
                print()
                response_text = FALLBACK_ACTION

            # Parse actions from response
            satellite_actions = parse_model_action(response_text)

            # Convert to correct format for environment
            action_dict = {int(k): v for k, v in satellite_actions.items()}

            print(f"\nStep {step}:")
            print(f"  Actions: {action_dict}")

            # Execute action
            action = SatelliteAction(satellite_actions=action_dict)
            observation = env.step(action)

            # Convert observation for display
            obs_dict = {
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

            step_reward = observation.reward or 0.0
            total_episode_reward += step_reward

            history_line = f"Step {step}: {action_dict} → reward {step_reward:+.2f}"
            history.append(history_line)

            print(f"  Reward: {step_reward:+.2f} | Total: {total_episode_reward:+.2f}")
            print(f"  Done: {observation.done}")

            if observation.done:
                print("\nEpisode complete.")
                break

        else:
            print(f"\nReached max steps ({MAX_STEPS}).")

        print(f"\n{'='*50}")
        print(f"Episode Summary:")
        print(f"  Total Reward: {total_episode_reward:.2f}")
        print(f"  Final Step: {obs_dict['time_step']}")
        print(f"  Satellites: {len(obs_dict['satellites'])}")
        for sat in obs_dict["satellites"]:
            print(
                f"    Satellite {sat['id']}: "
                f"Battery={sat['battery']:.1f}%, Storage={sat['storage']:.1f}%"
            )
        print(f"{'='*50}")

    finally:
        print("\nClosing environment...")


if __name__ == "__main__":
    main()
