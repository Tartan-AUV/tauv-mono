# ControllerDebug.msg  
# Debug information from the INDI Controller node

std_msgs/Header header

# Acceleration errors (body frame)
geometry_msgs/Vector3 linear_acceleration_error
geometry_msgs/Vector3 angular_acceleration_error

# Desired accelerations (from command)
geometry_msgs/Vector3 desired_linear_acceleration
geometry_msgs/Vector3 desired_angular_acceleration

# Measured/filtered accelerations (body frame)
geometry_msgs/Vector3 filtered_linear_acceleration
geometry_msgs/Vector3 filtered_angular_acceleration

# Control outputs
geometry_msgs/Wrench current_wrench
geometry_msgs/Wrench previous_wrench
geometry_msgs/Wrench wrench_increment

# Control effectiveness matrix condition number
float64 control_effectiveness_condition_number

# Filter parameter
float64 accel_filter_alpha

# Saturation flags
bool force_saturated
bool torque_saturated

# Magnitude information
float64 force_magnitude
float64 torque_magnitude
float64 max_force_limit
float64 max_torque_limit 