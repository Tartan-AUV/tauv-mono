#include "tauv_common/depth_converter.h"

DepthConverter::DepthConverter(std::string prefix) : Node("depth_converter"), prefix_(prefix) {
    sub_ = create_subscription<
        sensor_msgs::msg::FluidPressure>(prefix_ + "/sensors/pressure",
                                         rclcpp::SensorDataQoS(),
                                         std::bind(&DepthConverter::pressureCallback,
                                                   this,
                                                   std::placeholders::_1));

    pub_ = create_publisher<nav_msgs::msg::Odometry>(prefix_ + "/sensors/pressure/odom", 10);
}

void DepthConverter::pressureCallback(const sensor_msgs::msg::FluidPressure::SharedPtr msg) {
    nav_msgs::msg::Odometry odom;

    odom.header.stamp = msg->header.stamp;
    odom.header.frame_id = "odom";
    odom.child_frame_id = prefix_ + "/body_link";

    double pressure = msg->fluid_pressure;  // in Pascals
    // Simple linear approximation: depth (m) = (pressure (Pa) - atmospheric pressure
    double z = -(pressure - 101325.0) / 9806.65;  // assuming freshwater and g ~9.81 m/s²

    odom.pose.pose.position.z = z;
    odom.pose.pose.orientation.w = 1.0;

    odom.pose.covariance.fill(1000);
    odom.twist.covariance.fill(1000);

    odom.pose.covariance[14] = msg->variance;  // Z position variance

    pub_->publish(odom);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthConverter>("os");
    rclcpp::spin(node);
    rclcpp::shutdown();
}