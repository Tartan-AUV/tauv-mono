/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *
 *  Author:      root
 *  Date:        1/4/26
 *****************************************************************************/

#include "tauv_sim/context.h"

rclcpp::Time Context::get_ros_time() const { return rclcpp::Time(sim_time_.count()); }
