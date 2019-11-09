#pragma once
#include <geometry_msgs/Quaternion.h>
#include <tf/tf.h>

inline geometry_msgs::Quaternion Quaternion(double x, double y, double z,
                                            double w) {
  geometry_msgs::Quaternion msg;
  msg.x = x;
  msg.y = y;
  msg.z = z;
  msg.w = w;
  return msg;
}

inline geometry_msgs::Quaternion QuaternionRPY(double roll, double pitch,
                                               double yaw) {
  tf::Quaternion quat;
  quat.setEuler(yaw, pitch, roll);

  return Quaternion(quat.x(), quat.y(), quat.z(), quat.w());
}

inline geometry_msgs::Quaternion Quaternion() { return Quaternion(0, 0, 0, 0); }
