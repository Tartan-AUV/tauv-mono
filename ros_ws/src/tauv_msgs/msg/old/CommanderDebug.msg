# CommanderDebug.msg
# Debug information from the Commander node

std_msgs/Header header

# Control mode information
bool velocity_control_enabled
bool attitude_control_enabled
bool position_hold_enabled

# Position error (in world frame) [m] - only valid when position_hold_enabled
geometry_msgs/Vector3 position_error_world

# Velocity error (in body frame) [m/s] - only valid when velocity_control_enabled
geometry_msgs/Vector3 velocity_error_body

# Attitude error (as rotation vector) [rad] - only valid when attitude_control_enabled
geometry_msgs/Vector3 attitude_error_rotation_vector

# Angular velocity error (in body frame) [rad/s] - only valid when attitude_control_enabled
geometry_msgs/Vector3 angular_velocity_error_body

# Commanded accelerations (body frame)
geometry_msgs/Vector3 linear_acceleration_command
geometry_msgs/Vector3 angular_acceleration_command

# Current filtered state estimates
geometry_msgs/Vector3 filtered_velocity_body
geometry_msgs/Vector3 filtered_angular_velocity_body

# Gain values being used
float64 kp_velocity
float64 kd_velocity
float64 kp_attitude
float64 kd_attitude
float64 kp_position
float64 kd_position

# Command saturation flags
bool linear_accel_saturated
bool angular_accel_saturated 