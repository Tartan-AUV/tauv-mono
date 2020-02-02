#pragma once
#include <geometry_msgs/Pose.h>

#include "constructors/Point.h"
#include "constructors/Quaternion.h"

inline geometry_msgs::Pose Pose(geometry_msgs::Point position,
                                geometry_msgs::Quaternion orientation) {
  geometry_msgs::Pose msg;
  msg.position = position;
  msg.orientation = orientation;
  return msg;
}

inline geometry_msgs::Pose Pose() { return Pose(Point(), Quaternion()); }
