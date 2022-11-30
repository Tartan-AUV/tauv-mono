#include <iostream>
#include <ros/ros.h>
#include "state_estimator.h"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "state_estimation");
  ros::NodeHandle n("state_estimation");

  ROS_INFO("Launched!");

  StateEstimator s(n);

  while (ros::ok())
  {
    ros::spinOnce();
  }

  return 0;
}
