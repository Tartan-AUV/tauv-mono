#pragma once

#include <sensors/scalar/IMU.h>

#include <array>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <string>

#include "tauv_sim/config.h"
#include "tauv_sim/context.h"

class ImuBridge {
   public:
    ImuBridge(sf::IMU* sensor,
              rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub,
              std::string frame_id,
              const config::osprey::sensors::Imu& cfg);

    void on_step(const Context& ctx);

   private:
    sf::IMU* sensor_;
    std::string frame_id_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;

    std::array<double, 9> orientation_covariance_;
    std::array<double, 9> angular_velocity_covariance_;
    std::array<double, 9> linear_acceleration_covariance_;
};
