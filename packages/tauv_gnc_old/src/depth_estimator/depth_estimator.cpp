//
// Created by gleb on 5/22/25.
//

#include "tauv_gnc/depth_estimator/depth_estimator.hpp"

using std::placeholders::_1;

DepthEstimator::DepthEstimator() : Node("depth_estimator") {
  this->depth_sensor_frame_sub_ =
      this->create_subscription<tauv_msgs::msg::DepthSensorFrame>(
          "depth_sensor_frame", 10,
          std::bind(&DepthEstimator::depth_sensor_frame_callback, this, _1));
  this->reset_service_ = this->create_service<std_srvs::srv::Trigger>(
      "depth_sensor_frame",
      [this](std_srvs::srv::Trigger::Request::ConstSharedPtr request,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        this->reset_service_callback(request, response);
      });

  this->depth_pub_ = this->create_publisher<tauv_msgs::msg::Depth>("depth", 10);

  surface_pressure_ = declare_parameter<double>("surface_pressure", 101325.0);
  water_density_ = declare_parameter<double>("water_density", 997.0);
  gravity_ = declare_parameter<double>("gravity", 9.81);
  variance_ = declare_parameter<double>("variance", 1.0e-4);
}

void DepthEstimator::depth_sensor_frame_callback(
    const tauv_msgs::msg::DepthSensorFrame& msg) {
  if (reset_triggered_) {
    reset_triggered_ = false;
    surface_pressure_ = msg.pressure;
  }

  tauv_msgs::msg::Depth depth;
  depth.header = msg.header;
  depth.depth = (msg.pressure - surface_pressure_) / (water_density_ * gravity_);
  depth.variance = variance_;
  depth_pub_->publish(depth);
}

void DepthEstimator::reset_service_callback(
    std_srvs::srv::Trigger::Request::ConstSharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response) {
  reset_triggered_ = true;
  response->success = true;
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DepthEstimator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
