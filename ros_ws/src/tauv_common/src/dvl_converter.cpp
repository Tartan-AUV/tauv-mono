#include "tauv_common/dvl_converter.h"


DvlConverter::DvlConverter(std::string prefix) : Node("dvl_converter"), prefix_(prefix) {
    sub_ = create_subscription<
        tauv_msgs::msg::Dvl>(prefix_ + "/sensors/dvl",
                             rclcpp::SensorDataQoS(),
                             std::bind(&DvlConverter::dvlCallback,
                                       this,
                                       std::placeholders::_1));

    pub_ = create_publisher<nav_msgs::msg::Odometry>(prefix_ + "/sensors/dvl/odom", 10);
}

void DvlConverter::dvlCallback(const tauv_msgs::msg::Dvl::SharedPtr msg) {
    nav_msgs::msg::Odometry odom;

    odom.header.stamp = msg->header.stamp;
    odom.header.frame_id = "odom";
    odom.child_frame_id = prefix_ + "/body_link";

    // Convert DVL velocities to odometry twist
    odom.twist.twist.linear.x = msg->linear_velocity.x;
    odom.twist.twist.linear.y = msg->linear_velocity.y;
    odom.twist.twist.linear.z = msg->linear_velocity.z;

    odom.pose.pose.orientation.w = 1.0;  // No orientation info from DVL

    odom.pose.covariance.fill(1e6);
    odom.twist.covariance.fill(1e6);

    // Convert percent → fraction
    const double p = msg->linear_velocity_percent_noise * 0.01;
    const double sigma0 = msg->linear_velocity_stddev_noise;

    const double var_x = std::pow(p * msg->linear_velocity.x, 2) + std::pow(sigma0, 2);
    const double var_y = std::pow(p * msg->linear_velocity.y, 2) + std::pow(sigma0, 2);
    const double var_z = std::pow(p * msg->linear_velocity.z, 2) + std::pow(sigma0, 2);

    // Twist covariance indices (row-major 6x6)
    odom.twist.covariance[0]  = var_x;  // vx
    odom.twist.covariance[7]  = var_y;  // vy
    odom.twist.covariance[14] = var_z;  // vz;

    pub_->publish(odom);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DvlConverter>("os");
    rclcpp::spin(node);
    rclcpp::shutdown();
}