#pragma once
#include <std_msgs/Bool.h>

inline std_msgs::Bool Bool(bool data) {
  std_msgs::Bool val;
  val.data = data;
  return val;
}

inline std_msgs::Bool Bool() { return Bool(false); }
