# ROS Packages

This folder is the colcon workspace and home for all ROS 2 packages. Packages live in `src/`, and most are git submodules (see the [main README](../README.md#contributing) for the contribution workflow). Build and run instructions are in [docs/sim.md](../docs/sim.md).

## Conventions

### Package READMEs

Every package must have a `README.md` in its root folder explaining the higher-level goals of the package:
- What the package is for and where it fits in the system.
- The main nodes or launch files, and how to run them.
- The key topics, services, and parameters it exposes, at a summary level.
- Any setup, hardware, or dependency notes someone would need to know.

Keep it about the "why" and the overall design. Per-file details belong in the file headers below.

### Source file headers

Every source file (C++, Python, launch files, etc.) must start with a header comment that states:
- **Summary:** what the file does, in a few sentences.
- **Inputs:** what it consumes, usually ROS 2 topics (also services, actions, or parameters if relevant), with message types.
- **Outputs:** what it produces, usually ROS 2 topics (also services, actions, or files), with message types.

Python example:
```python
"""Estimates vehicle depth from the pressure sensor.

Inputs:
    /depth/pressure (sensor_msgs/FluidPressure): raw pressure readings
Outputs:
    /depth/depth (std_msgs/Float64): depth in meters, zeroed at the surface
"""
```

C++ example:
```cpp
/*
 * Summary: Estimates vehicle depth from the pressure sensor.
 *
 * Inputs:
 *   /depth/pressure (sensor_msgs/msg/FluidPressure): raw pressure readings
 * Outputs:
 *   /depth/depth (std_msgs/msg/Float64): depth in meters, zeroed at the surface
 */
```

Update the header whenever the topics or behavior change. If a file has no ROS inputs or outputs (a utility library, for example), say so and describe what it provides instead.
