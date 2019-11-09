#pragma once
#include <geometry_msgs/Vector3.h>

inline geometry_msgs::Vector3 Vector3(double x, double y, double z) {
  geometry_msgs::Vector3 val;
  val.x = x;
  val.y = y;
  val.z = z;
  return val;
}

inline geometry_msgs::Vector3 Vector3() { return Vector3(0, 0, 0); }
