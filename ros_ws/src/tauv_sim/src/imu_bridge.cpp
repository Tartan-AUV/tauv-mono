#include "tauv_sim/imu_bridge.h"

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

ImuBridge::ImuBridge(sf::IMU* sensor,
                     rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub,
                     std::string frame_id,
                     const config::osprey::sensors::Imu& cfg)
    : sensor_(sensor),
      frame_id_(std::move(frame_id)),
      pub_(pub),
      orientation_covariance_(diagonal_from_stddev(cfg.angle_std)),
      angular_velocity_covariance_(diagonal_from_stddev(cfg.angular_velocity_std)),
      linear_acceleration_covariance_(diagonal_from_stddev(cfg.linear_acceleration_std)) {}

void ImuBridge::on_step(const Context& ctx) {
    if (!sensor_->isNewDataAvailable()) {
        return;
    }

    const double roll = sensor_->getLastValue(0);
    const double pitch = sensor_->getLastValue(1);
    const double yaw = sensor_->getLastValue(2);

    const double ang_vel_x = sensor_->getLastValue(3);
    const double ang_vel_y = sensor_->getLastValue(4);
    const double ang_vel_z = sensor_->getLastValue(5);

    const double lin_accel_x = sensor_->getLastValue(6);
    const double lin_accel_y = sensor_->getLastValue(7);
    const double lin_accel_z = sensor_->getLastValue(8);

    Eigen::AngleAxisd roll_angle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_angle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond orientation = yaw_angle * pitch_angle * roll_angle;

    sensor_msgs::msg::Imu msg;
    msg.header.frame_id = frame_id_;
    msg.header.stamp = ctx.get_ros_time();

    msg.orientation.w = orientation.w();
    msg.orientation.x = orientation.x();
    msg.orientation.y = orientation.y();
    msg.orientation.z = orientation.z();
    std::copy(orientation_covariance_.begin(),
              orientation_covariance_.end(),
              msg.orientation_covariance.begin());

    msg.angular_velocity.x = ang_vel_x;
    msg.angular_velocity.y = ang_vel_y;
    msg.angular_velocity.z = ang_vel_z;
    std::copy(angular_velocity_covariance_.begin(),
              angular_velocity_covariance_.end(),
              msg.angular_velocity_covariance.begin());

    msg.linear_acceleration.x = lin_accel_x;
    msg.linear_acceleration.y = lin_accel_y;
    msg.linear_acceleration.z = lin_accel_z;
    std::copy(linear_acceleration_covariance_.begin(),
              linear_acceleration_covariance_.end(),
              msg.linear_acceleration_covariance.begin());

    pub_->publish(msg);
}
