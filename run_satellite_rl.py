import os
import random
import time
from typing import Dict

# Import the environment package. It loads .env automatically via its __init__.
from satellite_env import SatelliteConstellationEnv, Action, EasyTask

# Try to import OpenAI client used in archive/check_hf_token.py pattern
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def build_hf_client():
    """Create a HuggingFace router-backed OpenAI client if HF_TOKEN is present.

    Returns the client or None if not available.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or OpenAI is None:
        return None

    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)


def ask_llm_for_action(client, observation) -> Dict[int, str]:
    """Query the LLM to get actions for satellites.

    The prompt is intentionally simple: list each satellite id and desired action.
    If the LLM call fails, raise the exception to let caller handle fallback.
    """
    messages = [
        {"role": "system", "content": "You are an assistant that controls satellites. Reply with a JSON mapping of satellite_id to action (capture/downlink/maintain/idle)."},
        {"role": "user", "content": f"Observation: time_step={observation.time_step}, pending_tasks={observation.pending_tasks}. Provide actions for each satellite (ids: {[s.id for s in observation.satellites]})."}
    ]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        messages=messages,
        temperature=0.2,
        max_tokens=200,
    )

    text = completion.choices[0].message.get("content")
    # naive parse: expect something like {"0": "capture", "1": "idle"}
    import json

    try:
        parsed = json.loads(text)
        # convert keys to int
        return {int(k): v for k, v in parsed.items()}
    except Exception:
        # If direct JSON fails, try to extract lines like '0: capture'
        actions = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                left, right = line.split(':', 1)
                left = left.strip().strip('"')
                right = right.strip().strip('"')
                if left.isdigit():
                    actions[int(left)] = right
        if actions:
            return actions
        raise RuntimeError("Could not parse LLM response as actions")


def random_policy(observation) -> Dict[int, str]:
    choices = ['capture', 'downlink', 'maintain', 'idle']
    return {s.id: random.choice(choices) for s in observation.satellites}


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
            except Exception as e:
                print(f"LLM call failed: {e}; falling back to random policy.")
                action_map = random_policy(obs)
        else:
            action_map = random_policy(obs)

        action = Action(satellite_actions=action_map)
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
