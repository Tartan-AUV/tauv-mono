#include "controller.h"

#define TOPIC_QUEUE_SIZE 10

Controller::Controller(ros::NodeHandle &n) : n(n), alarm_client(n) {
  ROS_INFO("Controller initialized");

  this->load_config();

  this->timer = this->n.createTimer(this->dt, &Controller::update, this);

  this->acceleration.setZero();
  this->pose.setZero();
  this->twist.setZero();

  this->dynamics.set_m(1.2);
  this->pose(4) = M_PI / 4.0;
  this->acceleration(0) = 1.0;

  this->acceleration_sub = this->n.subscribe(
      "acceleration", TOPIC_QUEUE_SIZE, &Controller::handle_acceleration, this);

  this->state_sub = this->n.subscribe("state", TOPIC_QUEUE_SIZE,
                                      &Controller::handle_state, this);
}

void Controller::load_config() {
  int frequency;
  this->n.getParam("frequency", frequency);
  this->dt = ros::Duration(1.0 / double(frequency));
}

void Controller::update(const ros::TimerEvent &e) {
  Eigen::Matrix<double, 6, 1> wrench;
  this->dynamics.compute_tau(wrench, this->pose, this->twist,
                             this->acceleration);

  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  std::cout << wrench.transpose().format(clean_fmt) << std::endl;
}

void Controller::handle_acceleration(
    const tauv_msgs::ControllerCmd::ConstPtr &msg) {}

void Controller::handle_state(const tauv_msgs::Pose::ConstPtr &msg) {}