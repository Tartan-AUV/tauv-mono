#include "controller.h"
#include <ros/ros.h>

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "controller");
  ros::NodeHandle n("controller_cpp");

  ROS_INFO("Launched!");

  Controller c(n);

  while (ros::ok()) {
    ros::spinOnce();
  }

  return 0;
}