uint8 CONTROLLER=1
uint8 PLANNER=2

string planner

bool enableBuoyancyComp
bool enableVelocityFeedForward

uint8 vel_src_xy
uint8 vel_src_z
uint8 vel_src_heading
uint8 vel_src_attitude

uint8 acc_src_xy
uint8 acc_src_z
uint8 acc_src_heading
uint8 acc_src_attitude

# This setting is just for posting the status message
# Do not set when changing the controller settings
string[] available_planners