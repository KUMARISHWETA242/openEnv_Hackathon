#!/usr/bin/env python3
"""
Test script for the Satellite Constellation Environment
"""

from satellite_env import SatelliteConstellationEnv, Action, EasyTask, TaskGrader

def test_basic_functionality():
    """Test basic environment functionality"""
    print("Testing basic functionality...")

    env = SatelliteConstellationEnv(num_satellites=3, max_steps=10)

    # Test reset
    obs = env.reset()
    print(f"Reset successful. Satellites: {len(obs.satellites)}")

    # Test step
    action = Action(satellite_actions={0: "capture", 1: "maintain", 2: "idle"})
    obs, reward, done, info = env.step(action)
    print(f"Step successful. Reward: {reward.value}, Done: {done}")

    # Test state
    state = env.state()
    print(f"State retrieved. Time step: {state['time_step']}")

    print("Basic functionality test passed!")

def test_task_setup():
    """Test task setup and grading"""
    print("\nTesting task setup...")

    task = EasyTask()
    env = SatelliteConstellationEnv()
    task.setup_environment(env)

    grader = TaskGrader(task)
    criteria = task.get_success_criteria()
    print(f"Task: {task.name}")
    print(f"Criteria: {criteria}")

    # Run a short episode
    obs = env.reset()
    actions = []
    for _ in range(5):
        action = Action(satellite_actions={i: "capture" for i in range(env.num_satellites)})
        obs, reward, done, info = env.step(action)
        actions.append(action)
        if done:
            break

    final_state = env.state()
    score = grader.grade_episode(env, actions, final_state)
    print(f"Episode score: {score:.3f}")

    print("Task setup test passed!")

if __name__ == "__main__":
    test_basic_functionality()
    test_task_setup()
    print("\nAll tests passed! ✅")