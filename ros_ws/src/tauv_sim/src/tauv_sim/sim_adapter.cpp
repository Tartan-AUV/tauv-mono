//
// Created by gleb on 5/21/25.
//

#include "tauv_sim/sim_adapter.hpp"

using std::placeholders::_1;

SimAdapter::SimAdapter() : Node("sim_adapter") {

  dvl_subscription_ = this->create_subscription<stonefish_ros2::msg::DVL>(
      "sim/dvl", 10, std::bind(&SimAdapter::dvl_callback, this, _1));
  dvl_publisher_ =
      this->create_publisher<tauv_msgs::msg::WaterlinkedDvlFrame>("dvl", 10);
  pressure_subscription_ =
      this->create_subscription<sensor_msgs::msg::FluidPressure>(
          "sim/pressure", 10,
          std::bind(&SimAdapter::pressure_callback, this, _1));
  depth_publisher_ = this->create_publisher<tauv_msgs::msg::DepthSensorFrame>(
      "depth_sensor_frame", 10);
  thruster_setpoint_subscription_ =
      this->create_subscription<tauv_msgs::msg::RpmCommand>(
          "thruster_setpoint", 10,
          std::bind(&SimAdapter::thruster_setpoint_callback, this, _1));
  thruster_setpoint_publisher_ =
      this->create_publisher<std_msgs::msg::Float64MultiArray>(
          "sim/thruster_setpoint", 10);

  external_temperature_ = this->declare_parameter<double>("external_temperature", 25.0);
}

void SimAdapter::dvl_callback(const stonefish_ros2::msg::DVL& msg) {
  tauv_msgs::msg::WaterlinkedDvlFrame result;
  rclcpp::Time time_from_header(msg.header.stamp);

  result.header = msg.header;

  if (last_dvl_reading_stamp.has_value()) {
    result.time = (time_from_header - last_dvl_reading_stamp.value()).seconds();
  } else {
    result.time = 0.0f;
  }

  uint64_t unix_us_timestamp =
    static_cast<uint64_t>(time_from_header.nanoseconds() / 1000) +
    static_cast<uint64_t>(time_from_header.seconds()) * 1e6;

  result.time_of_transmission = unix_us_timestamp;
  result.time_of_validity = unix_us_timestamp;

  result.altitude = msg.altitude;

  result.vx = msg.velocity.x;
  result.vy = msg.velocity.y;
  result.vz = msg.velocity.z;

  result.velocity_valid = true;

  result.covariance = msg.velocity_covariance;

  result.status = 0;

  result.fom = 0.0f;

  dvl_publisher_->publish(result);
}

void SimAdapter::pressure_callback(const sensor_msgs::msg::FluidPressure& msg) {
  tauv_msgs::msg::DepthSensorFrame result;
  result.depth = -1.0f;
  result.pressure = msg.fluid_pressure;
  result.temperature = external_temperature_;
  result.header = msg.header;

  depth_publisher_->publish(result);
}

void SimAdapter::thruster_setpoint_callback(
    const tauv_msgs::msg::RpmCommand& msg) {
  std_msgs::msg::Float64MultiArray result;
  result.layout.dim.resize(1);
  result.layout.dim[0].label = "data";
  result.layout.dim[0].size = msg.enables.size();
  result.layout.dim[0].stride = msg.enables.size();
  result.layout.data_offset = 0;
  for (size_t i = 0; i < msg.enables.size(); ++i) {
    result.data.push_back(
      msg.enables[i] ? msg.rpms[i] * RPM_TO_RAD_PER_SEC : 0.0f);
  }

  thruster_setpoint_publisher_->publish(result);
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimAdapter>());
  rclcpp::shutdown();
  return 0;
}
