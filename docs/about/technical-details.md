# Technical Details

## Physical Simulation Model

### Orbital Mechanics

The environment implements simplified orbital mechanics suitable for reinforcement learning:

#### Satellite Movement
```python
# Position update each time step
new_position = (
    current_x + random_uniform(-10, 10),
    current_y + random_uniform(-10, 10),
    altitude  # Fixed altitude approximation
)
```

**Assumptions:**
- Circular orbits at ~400-600 km altitude
- Simplified 2D movement in orbital plane
- Random perturbations simulate orbital variations
- No gravitational effects or orbital decay

#### Orbital Parameters
- **Altitude Range**: 400-600 km (typical LEO)
- **Orbital Period**: Not explicitly modeled (time steps are abstract)
- **Ground Track**: Simplified linear movement with noise

### Resource Models

#### Battery System
- **Capacity**: 0-100% charge level
- **Base Drain**: 0.5% per time step (orbit maintenance)
- **Action Costs**:
  - Capture: -5%
  - Downlink: -2%
  - Maintain: +20% (charging)
  - Idle: 0%

#### Storage System
- **Capacity**: 0-100% utilization
- **Action Effects**:
  - Capture: +10% (image data)
  - Downlink: -20% (data transmission)
  - Maintain: 0%
  - Idle: 0%

#### Constraints
- Battery cannot go below 0%