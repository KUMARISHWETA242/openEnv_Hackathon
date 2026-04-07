# Task Specifications - Satellite Constellation Management

## Overview

The Satellite Constellation Management Environment includes three progressively difficult tasks that test different aspects of satellite operations management. Each task has specific objectives, constraints, and evaluation criteria.

## Task Hierarchy

### Difficulty Progression
1. **Easy**: Basic resource management and task execution
2. **Medium**: Multi-satellite coordination and data management
3. **Hard**: Large-scale constellation optimization with environmental constraints

## Easy Task: Basic Imaging

### Objective
Manage a small constellation of 3 satellites to capture Earth images while maintaining basic operational health.

### Configuration
- **Satellites**: 3
- **Max Steps**: 50
- **Weather**: Minimal impact
- **Communication**: Basic ground station access

### Initial Setup
- Satellites start with 100% battery and 0% storage
- Random orbital positions
- 5 pending imaging tasks across different regions

### Success Criteria
- **Min Images Captured**: 3
- **Min Final Battery**: 50% (average across satellites)
- **Max Steps**: 50

### Evaluation Function
```
score = 0.0
if images_captured >= 3:
    score += 1.0
else:
    score += images_captured / 3.0

if avg_final_battery >= 50:
    score += 1.0
else:
    score += avg_final_battery / 50.0

if steps > 50:
    score *= 0.8  # Penalty for exceeding time limit

final_score = min(1.0, score / 2.0)
```

### Key Challenges
- Balance image capture with battery management
- Learn basic action effects
- Handle simple resource constraints

### Expected Performance
- **Baseline Score**: 0.7
- **Human Expert**: 0.9+
- **Optimal**: 1.0

## Medium Task: Data Management

### Objective
Coordinate 5 satellites to capture images and manage data downlink operations while maintaining constellation health.

### Configuration
- **Satellites**: 5
- **Max Steps**: 100
- **Weather**: Moderate regional variations
- **Communication**: Multiple ground stations with realistic coverage

### Initial Setup
- Satellites start with 100% battery and 0% storage
- Varied orbital positions for coverage diversity
- 10 mixed tasks: imaging and data downlink requests

### Success Criteria
- **Min Images Captured**: 5
- **Min Data Downlinked**: 50 units
- **Min Final Battery**: 30% (average across satellites)
- **Max Steps**: 100

### Evaluation Function
```
score = 0.0
if images_captured >= 5:
    score += 1.0
else:
    score += images_captured / 5.0

if data_downlinked >= 50:
    score += 1.0
else:
    score += data_downlinked / 50.0

if avg_final_battery >= 30:
    score += 1.0
else:
    score += avg_final_battery / 30.0

if steps > 100:
    score *= 0.8

final_score = min(1.0, score / 3.0)
```

### Key Challenges
- Coordinate multiple satellites simultaneously
- Balance imaging vs. downlink priorities
- Manage communication windows
- Handle increased complexity

### Expected Performance
- **Baseline Score**: 0.5
- **Human Expert**: 0.8+
- **Optimal**: 1.0

## Hard Task: Constellation Coordination

### Objective
Optimize operations of an 8-satellite constellation under complex environmental conditions and competing objectives.

### Configuration
- **Satellites**: 8
- **Max Steps**: 200
- **Weather**: Significant regional variations (0.2-0.8 cloud cover)
- **Communication**: Multiple ground stations with limited coverage areas

### Initial Setup
- Satellites start with 100% battery and 0% storage
- Complex orbital configuration for global coverage
- 20 diverse tasks across multiple regions with varying priorities

### Success Criteria
- **Min Images Captured**: 10
- **Min Data Downlinked**: 100 units
- **Min Final Battery**: 20% (average across satellites)
- **Max Steps**: 200

### Evaluation Function
```
score = 0.0
if images_captured >= 10:
    score += 1.0
else:
    score += images_captured / 10.0

if data_downlinked >= 100:
    score += 1.0
else:
    score += data_downlinked / 100.0

if avg_final_battery >= 20:
    score += 1.0
else:
    score += avg_final_battery / 20.0

if steps > 200:
    score *= 0.8

final_score = min(1.0, score / 3.0)
```

### Key Challenges
- Large-scale coordination across 8 satellites
- Weather-dependent imaging opportunities
- Complex communication scheduling
- Long-term resource planning
- Competing task priorities

### Expected Performance
- **Baseline Score**: 0.3
- **Human Expert**: 0.7+
- **Optimal**: 1.0

## Task Characteristics

### Progressive Difficulty
| Aspect | Easy | Medium | Hard |
|--------|------|--------|------|
| Satellites | 3 | 5 | 8 |
| Max Steps | 50 | 100 | 200 |
| Tasks | 5 | 10 | 20 |
| Weather Complexity | Low | Medium | High |
| Communication Constraints | Basic | Moderate | Complex |

### Evaluation Metrics
All tasks are evaluated on:
1. **Task Completion**: Achievement of primary objectives
2. **Resource Efficiency**: Battery and storage management
3. **Time Efficiency**: Completion within step limits

### Scoring Methodology
- **Partial Credit**: Tasks reward proportional achievement of objectives
- **Time Penalties**: Exceeding step limits reduces final score
- **Normalization**: All scores scaled to 0.0-1.0 range
- **Deterministic**: Same actions always produce same scores

## Task Design Principles

### Realism
- Based on actual satellite operations challenges
- Realistic resource constraints and orbital mechanics
- Practical communication and weather considerations

### Learnability
- Clear reward signals for different actions
- Progressive complexity across tasks
- Intuitive cause-and-effect relationships

### Scalability
- Tasks designed to test different scales of coordination
- Modular design allows easy addition of new tasks
- Performance metrics enable comparison across difficulty levels

## Implementation Details

### Task Initialization
Each task modifies the base environment:
- Satellite count and positioning
- Task queue population
- Weather condition settings
- Time step limits

### Runtime Evaluation
Tasks are evaluated post-episode using:
- Action history analysis
- Final state inspection
- Success criteria checking
- Score computation and normalization

### Extension Points
Tasks can be extended by:
- Adding new task types
- Modifying success criteria
- Adjusting environmental parameters
- Creating domain-specific evaluation metrics