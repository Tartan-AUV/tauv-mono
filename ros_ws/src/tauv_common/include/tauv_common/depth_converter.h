#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <string>

class DepthConverter : public rclcpp::Node {
   public:
    DepthConverter(std::string prefix);

   private:
    void pressureCallback(const sensor_msgs::msg::FluidPressure::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::FluidPressure>::SharedPtr sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;

    std::string prefix_;
};
