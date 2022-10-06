#include "dynamics.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/IO.h>
#include <iostream>
#include <math.h>
#include <ros/ros.h>
#include <tauv_alarms/alarm_client.h>
#include <tauv_alarms/alarms.h>
#include <tauv_msgs/ControllerCmd.h>
#include <tauv_msgs/Pose.h>

#pragma once

class Controller {
public:
  Controller(ros::NodeHandle &n);

private:
  ros::NodeHandle &n;
  ros::Timer timer;

  tauv_alarms::AlarmClient alarm_client;

  ros::Duration dt;

  ros::Subscriber acceleration_sub;
  ros::Subscriber state_sub;

  Dynamics dynamics;

  Eigen::Matrix<double, 6, 1> acceleration;
  Eigen::Matrix<double, 6, 1> pose;
  Eigen::Matrix<double, 6, 1> twist;

  void load_config();

  void update(const ros::TimerEvent &e);

  void handle_acceleration(const tauv_msgs::ControllerCmd::ConstPtr &msg);

  void handle_state(const tauv_msgs::Pose::ConstPtr &msg);
};