#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tauv_msgs/msg/dvl.hpp>
#include <string>

class DvlConverter : public rclcpp::Node {
    public:
        DvlConverter(std::string prefix);

    private:
        void dvlCallback(const tauv_msgs::msg::Dvl::SharedPtr msg);

        rclcpp::Subscription<tauv_msgs::msg::Dvl>::SharedPtr sub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;

        std::string prefix_;
};
