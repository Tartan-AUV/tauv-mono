#pragma once
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <cstdint>
#include <string>

// Value constructor
inline std_msgs::Header Header(uint32_t seq, ros::Time timestamp,
                               std::string frame_id) {
  std_msgs::Header msg;
  msg.seq = seq;
  msg.stamp = timestamp;
  msg.frame_id = frame_id;
  return msg;
}

// Zero constructor
inline std_msgs::Header Header() { return Header(0, ros::Time(), ""); }
