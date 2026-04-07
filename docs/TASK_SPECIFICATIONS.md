# Task Specifications - Satellite Constellation Management

## Overview

The environment ships with three deterministic task presets. Each preset increases coordination cost, workload, and grading strictness.

## Difficulty Progression

| Task | Satellites | Max Steps | Pending Tasks | Seed | Added Challenge |
|------|------------|-----------|---------------|------|-----------------|
| Easy | 3 | 50 | 5 image captures | 101 | Basic battery and storage discipline |
| Medium | 5 | 100 | 12 mixed tasks | 202 | Capture/downlink balancing |
| Hard | 8 | 200 | 20 mixed tasks | 303 | Heavier cloud cover, more assets, stricter penalties |

## Easy Task

Objective:
- complete simple image workload without draining fleet health

Configuration:
- 3 satellites
- 50-step budget
- 5 image tasks in `region1`
- light weather pressure

Success criteria:
- at least 3 image tasks completed
- at least 3 total tasks completed
- average final battery at least 50
- invalid actions at most 3

## Medium Task

Objective:
- coordinate imaging and downlink flow while preserving usable battery

Configuration:
- 5 satellites
- 100-step budget
- 6 image tasks plus 6 downlink tasks
- each downlink task starts with 10 required units

Success criteria:
- at least 5 image tasks completed
- at least 50 downlinked units
- at least 8 total tasks completed
- average final battery at least 30
- invalid actions at most 8

## Hard Task

Objective:
- manage a larger fleet through mixed workload and weather-sensitive imaging

Configuration:
- 8 satellites
- 200-step budget
- 12 image tasks plus 8 downlink tasks
- `region1` has higher cloud cover than `region2`
- each downlink task starts with 12.5 required units

Success criteria:
- at least 10 image tasks completed
- at least 100 downlinked units
- at least 14 total tasks completed
- average final battery at least 20
- invalid actions at most 12

## Deterministic Grading

`TaskGrader` scores each episode on a `0.0` to `1.0` scale using:
- completed image tasks
- completed downlink workload
- total tasks completed
- final average battery
- invalid-action rate

The final score is the average of the applicable criterion scores, with a time-limit penalty if the episode exceeds the task step budget.

## Reward Shaping

The runtime reward is not the same as the final task score.

The environment gives intermediate feedback for:
- successful capture task completion
- successful downlink work and full downlink-task completion
- timely maintenance
- controlled low-battery recovery

It penalizes:
- invalid actions
- wasteful repeated actions
- unnecessary maintenance
- unproductive idling
- risky low-battery or near-full-storage states

## Why The Progression Matters

- `easy` teaches basic action effects and simple resource balancing
- `medium` forces agents to trade off storage growth against downlink demand
- `hard` requires longer-horizon coordination because the fleet is larger, the task list is longer, and weather makes naive capture policies less effective
