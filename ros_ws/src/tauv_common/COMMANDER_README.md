# AUV Commander System

The commander system provides high-level velocity and attitude control for the AUV by generating acceleration commands for the INDI controller. It operates as the outer loop in a cascaded control architecture.

## Architecture

```
High-Level Mission Control
    ↓ (VelocityAttitudeCommand)
Commander Node
    ↓ (AccelStamped)  
INDI Controller
    ↓ (WrenchStamped)
Thruster Allocation
    ↓ (TargetThrust)
Vehicle Actuators
```

## Components

### 1. VelocityAttitudeCommand Message

```bash
# Target linear velocity in body frame [m/s]
geometry_msgs/Vector3 target_velocity

# Target attitude as quaternion (orientation in world frame)  
geometry_msgs/Quaternion target_attitude

# Optional feedforward acceleration (if known)
geometry_msgs/Vector3 feedforward_acceleration

# Control enable flags
bool velocity_control_enabled
bool attitude_control_enabled
```

### 2. Commander Node

**Subscribes to:**
- `/gnc/velocity_attitude_command` (VelocityAttitudeCommand): Target velocity and attitude
- `/gnc/navigation_state` (NavigationState): Current vehicle state

**Publishes to:**
- `/gnc/acceleration_command` (AccelStamped): Acceleration commands for INDI controller

**Control Laws:**
- **Velocity Control**: Proportional-derivative control using velocity error
- **Attitude Control**: Quaternion-based control with rotation vector error representation

### 3. INDI Controller Node

The existing INDI controller accepts acceleration commands and produces wrench commands using incremental nonlinear dynamic inversion.

## Usage

### Running the System

1. **Start the INDI controller:**
   ```bash
   ros2 run tauv_common indi_controller
   ```

2. **Start the commander:**
   ```bash
   ros2 run tauv_common commander
   ```

3. **Send commands using the example:**
   ```bash
   ros2 run tauv_common commander_example
   ```

### Command Examples

**Forward velocity command:**
```python
cmd = VelocityAttitudeCommand()
cmd.target_velocity.x = 0.5  # 0.5 m/s forward
cmd.velocity_control_enabled = True
cmd.attitude_control_enabled = False
publisher.publish(cmd)
```

**Attitude command (30° yaw turn):**
```python
yaw_angle = np.deg2rad(30)
target_quat = UnitQuaternion.Rz(yaw_angle)

cmd = VelocityAttitudeCommand()
cmd.target_attitude.w = target_quat.q[0]
cmd.target_attitude.x = target_quat.q[1] 
cmd.target_attitude.y = target_quat.q[2]
cmd.target_attitude.z = target_quat.q[3]
cmd.velocity_control_enabled = False
cmd.attitude_control_enabled = True
publisher.publish(cmd)
```

**Combined velocity and attitude control:**
```python
cmd = VelocityAttitudeCommand()
cmd.target_velocity.x = 0.3  # Forward motion
cmd.target_attitude = target_quaternion  # Desired orientation
cmd.velocity_control_enabled = True
cmd.attitude_control_enabled = True
publisher.publish(cmd)
```

### Parameter Tuning

The commander uses configurable control gains in `CommanderParams`:

```python
# Velocity control gains
kp_velocity: float = 2.0        # Proportional gain [1/s]
kd_velocity: float = 0.1        # Derivative gain [1]

# Attitude control gains  
kp_attitude: float = 1.5        # Proportional gain [1/s²]
kd_attitude: float = 0.3        # Derivative gain [1/s]

# Control limits
max_linear_accel: float = 2.0   # Max acceleration [m/s²]
max_angular_accel: float = 1.0  # Max angular accel [rad/s²]
```

### Coordinate Frames

- **Velocity commands**: Body frame (x=forward, y=left, z=up)
- **Attitude commands**: World frame quaternions
- **Acceleration outputs**: Body frame

## Implementation Notes

### Velocity Control
- Uses proportional control on velocity error: `a_cmd = Kp * (v_desired - v_current)`
- Includes derivative term for improved performance when previous measurements are available
- Supports feedforward acceleration terms

### Attitude Control
- Computes quaternion error: `q_error = q_target * q_current^(-1)`
- Converts to rotation vector using logarithmic map for small angles
- Uses PD structure: `alpha_cmd = Kp * rotation_error + Kd * angular_velocity_error`

### Safety Features
- Acceleration magnitude limiting
- Individual enable flags for velocity and attitude control
- Velocity filtering for derivative estimation
- Graceful handling of missing inputs

## Testing

The `commander_example` node provides a demonstration sequence:
1. Forward velocity command
2. Yaw turn maneuver
3. Combined velocity and attitude control
4. Return to center heading
5. Emergency stop

Run `ros2 run tauv_common commander_example` to see the system in action.

## Integration with INDI Theory

This implementation follows INDI principles by:
- Operating in the outer loop to provide acceleration references
- Using current state feedback for error computation
- Generating smooth acceleration commands compatible with INDI's incremental structure
- Supporting feedforward terms for improved tracking performance

The commander provides the desired accelerations that INDI uses as reference signals, completing the cascaded control architecture recommended for underwater vehicles. 