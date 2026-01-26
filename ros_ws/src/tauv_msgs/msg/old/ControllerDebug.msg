# ControllerDebug.msg  
# Debug information from the INDI Controller node

std_msgs/Header header

# Outer loop: velocity error
geometry_msgs/Vector3 linear_velocity_error
geometry_msgs/Vector3 angular_velocity_error

# Measured/filtered accelerations (body frame)
geometry_msgs/Vector3 filtered_linear_acceleration
geometry_msgs/Vector3 filtered_angular_acceleration

# Desired accelerations accelerations (from outer lop)
geometry_msgs/Vector3 desired_linear_acceleration
geometry_msgs/Vector3 desired_angular_acceleration

# Acceleration errors (body frame)
geometry_msgs/Vector3 linear_acceleration_error
geometry_msgs/Vector3 angular_acceleration_error

# Control outputs
geometry_msgs/Wrench wrench_increment
