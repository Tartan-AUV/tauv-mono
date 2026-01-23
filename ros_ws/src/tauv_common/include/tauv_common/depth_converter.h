#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include "tauv_msgs/msg/pressure.hpp"

class DepthConverter : public rclcpp::Node{
    public:
        DepthConverter(std::string prefix, double z_stddev);

    private:
        void pressureCallback(const tauv_msgs::msg::Pressure::SharedPtr msg);

        rclcpp::Subscription<tauv_msgs::msg::Pressure>::SharedPtr sub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;

        std::string prefix_;

        double z_stddev_;
};
