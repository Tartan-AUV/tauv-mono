# TAUV Sim

## Frames of reference

**Vehicle body frame** - centered around the bottom-most point of the body

**Vehicle inertial frame** - origin is on CoM of the hull + all static attachments,
axes aligned with principal axes of inertia, such that the inertia matrix is diagonal

## Sim model

- Single link for the main hull
- Overriding both inertial and buyancy properties

## Kinematic mode

Run `tauv_sim` with `--kinematic path/to/trajectory.yaml` to play back a predefined body pose
trajectory without physics or thruster control. Trajectories are specified as a list of
keyframes:

```yaml
playback_mode: repeat  # onetime|repeat|boomerang (optional, defaults to onetime)
keyframes:
  - t: 0.0
    position: [0.0, 0.0, -1.0]
    quaternion: [0.0, 0.0, 0.0, 1.0]  # [x, y, z, w]
  - t: 5.0
    position: [2.0, 0.0, -1.0]
    quaternion: [0.0, 0.0, 0.0, 1.0]
```

Pass `--no-cameras` (or `--headless`) to skip creating the fisheye camera sensors and their ROS
publishers when running in headless or non-visual modes.
