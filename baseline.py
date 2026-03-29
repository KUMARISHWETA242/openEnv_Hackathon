#!/usr/bin/env python3
"""
Baseline inference script for Satellite Constellation Environment
Uses OpenAI API to run a model against the environment and produce scores.
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from satellite_env import SatelliteConstellationEnv, Action, EasyTask, MediumTask, HardTask, TaskGrader

def get_openai_action(client: OpenAI, env_state: Dict[str, Any], task_description: str) -> Action:
    """Get action from OpenAI model based on current state"""

    prompt = f"""
You are controlling a satellite constellation. Your goal: {task_description}

Current State:
{json.dumps(env_state, indent=2)}

Available actions for each satellite: 'capture', 'downlink', 'maintain', 'idle'

Respond with a JSON object containing satellite_actions as a dict of satellite_id -> action.
Example: {{"satellite_actions": {{"0": "capture", "1": "downlink"}}}}

Choose actions to maximize mission value.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content.strip())
        return Action(**result)
    except Exception as e:
        print(f"Error getting action: {e}")
        # Fallback to idle actions
        return Action(satellite_actions={i: "idle" for i in range(env_state['satellites'].__len__())})

def run_baseline(task: Any, task_name: str) -> float:
    """Run baseline for a specific task"""

    # Initialize environment and task
    env = SatelliteConstellationEnv()
    task.setup_environment(env)
    grader = TaskGrader(task)

    # OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Reset environment
    obs = env.reset()
    actions_taken = []
    done = False

    while not done:
        # Get current state
        state = env.state()

        # Get action from OpenAI
        action = get_openai_action(client, state, task.description)

        # Step environment
        obs, reward, done, info = env.step(action)
        actions_taken.append(action)

        if done:
            break

    # Grade the episode
    final_state = env.state()
    score = grader.grade_episode(env, actions_taken, final_state)

    print(f"{task_name} Score: {score:.3f}")
    return score

def main():
    """Run baseline on all tasks"""

    tasks = [
        (EasyTask(), "Easy"),
        (MediumTask(), "Medium"),
        (HardTask(), "Hard")
    ]

    results = {}
    for task, name in tasks:
        try:
            score = run_baseline(task, name)
            results[name.lower()] = score
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name.lower()] = 0.0

    print("\nFinal Results:")
    for task, score in results.items():
        print(f"{task}: {score:.3f}")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()