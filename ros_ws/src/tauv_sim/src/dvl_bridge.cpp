#include "tauv_sim/dvl_bridge.h"

#include <Eigen/Geometry>
#include <algorithm>

namespace {

std::array<double, 9> diagonal_from_stddev(const sf::Vector3& stddev) {
    const double sx = static_cast<double>(stddev.x());
    const double sy = static_cast<double>(stddev.y());
    const double sz = static_cast<double>(stddev.z());

    return {sx * sx, 0.0, 0.0, 0.0, sy * sy, 0.0, 0.0, 0.0, sz * sz};
}

}  // namespace

DvlBridge::DvlBridge(sf::DVL* sensor,
                     rclcpp::Publisher<tauv_msgs::msg::Dvl>::SharedPtr pub,
                     std::string frame_id,
                     const config::osprey::sensors::Dvl& cfg)
    : sensor_(sensor),
      frame_id_(std::move(frame_id)),
      pub_(pub),
      linear_velocity_percent_noise_(cfg.linear_velocity_percent_noise),
      linear_velocity_stddev_noise_(cfg.linear_velocity_stddev_noise) {}

void DvlBridge::on_step(const Context& ctx) {
    if (!sensor_->isNewDataAvailable()) {
        return;
    }

    const double lin_vel_x = sensor_->getLastValue(0);
    const double lin_vel_y = sensor_->getLastValue(1);
    const double lin_vel_z = sensor_->getLastValue(2);

    tauv_msgs::msg::Dvl msg;
    msg.header.frame_id = frame_id_;
    msg.header.stamp = ctx.get_ros_time();

    msg.linear_velocity.x = lin_vel_x;
    msg.linear_velocity.y = lin_vel_y;
    msg.linear_velocity.z = lin_vel_z;

    msg.linear_velocity_percent_noise = linear_velocity_percent_noise_;
    msg.linear_velocity_stddev_noise = linear_velocity_stddev_noise_;

    pub_->publish(msg);
}
