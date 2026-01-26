#pragma once

#include <sensors/scalar/Pressure.h>

#include <sensor_msgs/msg/fluid_pressure.hpp>

#include "tauv_sim/context.h"

class PressureSensorBridge {
   public:
    PressureSensorBridge(sf::Pressure* sensor,
                         rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr pub,
                         std::string frame_id);

    void on_step(const Context& ctx);

   private:
    sf::Pressure* sensor_pressure_;
    const std::string frame_id_;
    rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr pub_;
};
