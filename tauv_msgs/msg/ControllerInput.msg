# These must be w.r.t. the odom frame
Header header

# Enables position control for each channel: [x,y,z,r,p,y]
bool[] enable_pos_control

# Enable velocity control for each channel [x,y,z,r,p,y]
bool[] enable_vel_control

# Position targets in the odom frame. [x,y,z,qw,qx,qy,qz]
float64[] pos_target

# Velocity targets in the odom frame. [x,y,z,wx,wy,wz]
float64[] vel_target
# If set, use angular velocity in the inertial frame insted of odom.
bool use_inertial_ang_vel

# Feed-forward torque : [x,y,z,tx,ty,tz]. Linear forces are in odom, torques in base_link.
float64[] torque_ff
