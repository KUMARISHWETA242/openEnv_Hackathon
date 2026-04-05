# Contributing

## Contributing Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for all function parameters and return values
- Write comprehensive docstrings
- Keep functions focused and modular

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for complex features
- Performance benchmarks for optimizations
- Documentation updates for API changes

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request with detailed description

### Issue Reporting
- Use GitHub issues for bug reports and feature requests
- Include environment details and reproduction steps
- Provide example code when possible

## Performance Optimization

### Profiling
```python
import cProfile
import pstats

def profile_environment():
    env = SatelliteConstellationEnv(num_satellites=50)

    profiler = cProfile.Profile()
    profiler.enable()

    obs = env.reset()
    for _ in range(100):
        action = Action(satellite_actions={i: 'idle' for i in range(50)})
        obs, reward, done, info = env.step(action)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Optimization Techniques
1. **Vectorization**: Use NumPy for batch operations
2. **Caching**: Cache expensive calculations
3. **Lazy Evaluation**: Compute values only when needed
4. **Memory Pooling**: Reuse object instances