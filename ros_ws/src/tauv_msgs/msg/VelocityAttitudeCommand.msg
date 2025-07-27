# VelocityAttitudeCommand.msg
# Command message for specifying desired velocity and attitude for the AUV

std_msgs/Header header

# Target linear velocity in body frame [m/s]
geometry_msgs/Vector3 target_velocity

# Target attitude as quaternion (orientation in world frame)
geometry_msgs/Quaternion target_attitude

# Optional feedforward acceleration (if known)
geometry_msgs/Vector3 feedforward_acceleration

# Control enable flags
bool velocity_control_enabled
bool attitude_control_enabled 