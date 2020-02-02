#pragma once
#include <geometry_msgs/PoseStamped.h>

#include "Header.h"
#include "Pose.h"

inline geometry_msgs::PoseStamped PoseStamped(std_msgs::Header header,
                                              geometry_msgs::Pose pose) {
  geometry_msgs::PoseStamped msg;
  msg.header = header;
  msg.pose = pose;
  return msg;
}

inline geometry_msgs::PoseStamped PoseStamped() {
  return PoseStamped(Header(), Pose());
}
