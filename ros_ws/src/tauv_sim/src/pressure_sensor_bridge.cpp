#include "tauv_sim/pressure_sensor_bridge.h"

#include <sensors/scalar/Pressure.h>

PressureSensorBridge::PressureSensorBridge(
    sf::Pressure* sensor,
    rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr pub,
    std::string frame_id)
    : sensor_pressure_(sensor), frame_id_(std::move(frame_id)), pub_(pub) {}

void PressureSensorBridge::on_step(const Context& ctx) {
    if (sensor_pressure_->isNewDataAvailable()) {
        const float pressure = sensor_pressure_->getLastValue(0);

        sensor_msgs::msg::FluidPressure pressure_msg;
        pressure_msg.fluid_pressure = pressure;
        pressure_msg.variance = 0.0;  // Variance can be set if noise characteristics are known
        pressure_msg.header.frame_id = frame_id_;
        pressure_msg.header.stamp = ctx.get_ros_time();
        pub_->publish(pressure_msg);
    }
}
