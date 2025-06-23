//
// Created by gleb on 5/22/25.
//

#pragma once

#include <rclcpp/rclcpp.hpp>

#include <tauv_msgs/msg/depth_sensor_frame.hpp>
#include <tauv_msgs/msg/depth.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>

class DepthEstimator final : public rclcpp::Node {
public:
  DepthEstimator();

private:
  rclcpp::Subscription<tauv_msgs::msg::DepthSensorFrame>::SharedPtr depth_sensor_frame_sub_;
  rclcpp::Publisher<tauv_msgs::msg::Depth>::SharedPtr depth_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;

  void depth_sensor_frame_callback(const tauv_msgs::msg::DepthSensorFrame &msg);
  void reset_service_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr request,
                              std_srvs::srv::Trigger::Response::SharedPtr response);

  double surface_pressure_;
  double water_density_;
  double gravity_;
  double variance_;

  bool reset_triggered_ = false;
};
