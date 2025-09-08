//
// Created by gleb on 5/21/25.
//

#ifndef SIM_ADAPTER_H
#define SIM_ADAPTER_H

#include "rclcpp/rclcpp.hpp"

#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/fluid_pressure.hpp"
#include "tauv_msgs/msg/depth_sensor_frame.hpp"
#include "tauv_msgs/msg/waterlinked_dvl_frame.hpp"
#include "tauv_msgs/msg/thruster_setpoint.hpp"
#include "stonefish_ros2/msg/dvl.hpp"

class SimAdapter final : public rclcpp::Node {
public:
  SimAdapter();

private:
  // Sensors
  rclcpp::Subscription<stonefish_ros2::msg::DVL>::SharedPtr dvl_subscription_;
  rclcpp::Publisher<tauv_msgs::msg::WaterlinkedDvlFrame>::SharedPtr dvl_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::FluidPressure>::SharedPtr pressure_subscription_;
  rclcpp::Publisher<tauv_msgs::msg::DepthSensorFrame>::SharedPtr depth_publisher_;

  // Actuators
  rclcpp::Subscription<tauv_msgs::msg::ThrusterSetpoint>::SharedPtr thruster_setpoint_subscription_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr thruster_setpoint_publisher_;
  rclcpp::TimerBase::SharedPtr thruster_timeout_timer_;

  // Callbacks
  void dvl_callback(const stonefish_ros2::msg::DVL &msg);
  void pressure_callback(const sensor_msgs::msg::FluidPressure &msg);
  void thruster_setpoint_callback(const tauv_msgs::msg::ThrusterSetpoint &msg);
  void thruster_timeout_callback();

  // State variables
  std::optional<rclcpp::Time> last_dvl_reading_stamp;
  rclcpp::Time last_thruster_command_time_;
  size_t num_thrusters_;

  // Constants
  double external_temperature_;
  double thruster_max_rpm_;

  static constexpr double PI = 3.14159265358979323846;
  static constexpr double RPM_TO_RAD_PER_SEC = 1.0f / 60.0f * PI;
  static constexpr std::chrono::milliseconds THRUSTER_TIMEOUT{30};
};



#endif //SIM_ADAPTER_H
