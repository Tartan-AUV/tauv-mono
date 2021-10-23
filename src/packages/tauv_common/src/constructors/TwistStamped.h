#pragma once
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Header.h>

#include "constructors/Header.h"
#include "constructors/Twist.h"

inline geometry_msgs::TwistStamped TwistStamped(std_msgs::Header header,
                                                geometry_msgs::Twist twist) {
  geometry_msgs::TwistStamped msg;
  msg.header = header;
  msg.twist = twist;
  return msg;
}

inline geometry_msgs::TwistStamped TwistStamped() {
  return TwistStamped(Header(), Twist());
}
