#include "tauv_sim/pressure_sensor_bridge.h"

#include <sensors/scalar/Pressure.h>

PressureSensorBridge::PressureSensorBridge(
    sf::Pressure* sensor,
    rclcpp::Publisher<tauv_msgs::msg::Pressure>::SharedPtr pub,
    std::string frame_id)
    : sensor_(sensor), frame_id_(std::move(frame_id)), pub_(pub) {}

void PressureSensorBridge::on_step(const Context& ctx) {
    if (sensor_->isNewDataAvailable()) {
        const float pressure = sensor_->getLastValue(0);

        tauv_msgs::msg::Pressure pressure_msg;
        pressure_msg.pressure = pressure;
        pressure_msg.avg_window = 0.0F;
        pressure_msg.header.frame_id = frame_id_;
        pressure_msg.header.stamp = ctx.get_ros_time();

        pub_->publish(pressure_msg);
    }
}
