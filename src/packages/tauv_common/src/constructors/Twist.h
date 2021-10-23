#pragma once
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Vector3.h>

#include "Vector3.h"

inline geometry_msgs::Twist Twist(geometry_msgs::Vector3 linear,
                                  geometry_msgs::Vector3 angular) {
  geometry_msgs::Twist msg;
  msg.linear = linear;
  msg.angular = angular;
  return msg;
}

inline geometry_msgs::Twist Twist() { return Twist(Vector3(), Vector3()); }
