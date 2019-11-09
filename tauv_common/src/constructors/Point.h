#pragma once
#include <geometry_msgs/Point.h>

inline geometry_msgs::Point Point(double x, double y, double z) {
  geometry_msgs::Point msg;
  msg.x = x;
  msg.y = y;
  msg.z = z;
  return msg;
}

inline geometry_msgs::Point Point() { return Point(0, 0, 0); }
