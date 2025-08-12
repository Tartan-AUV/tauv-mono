//
// Created by gleb on 5/18/25.
//

#ifndef RTVC_H
#define RTVC_H

#include <array>
#include <boost/asio.hpp>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>
#include <rclcpp/rclcpp.hpp>

#include "flatbuffers/flatbuffers.h"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "sensor_msgs/msg/fluid_pressure.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/temperature.hpp"
#include "std_msgs/msg/string.hpp"
#include "tauv_msgs/msg/thruster_setpoint.hpp"
#include "tauv_msgs/msg/esc_telemetry.hpp"
#include "tauv_msgs/msg/depth_sensor_frame.hpp"
#include "tauv_vehicle/generated/eth_msg_jetson_rtvc_50_generated.h"
#include "tauv_vehicle/generated/eth_msg_rtvc_jetson_100_generated.h"
#include "tauv_vehicle/generated/eth_msg_rtvc_jetson_50_generated.h"

using boost::asio::ip::udp;
using namespace TAUV_FB;

class RTVCNode : public rclcpp::Node {
 public:
  RTVCNode();
  ~RTVCNode() override;

 private:
  struct XsensROSMessages {
    std::optional<sensor_msgs::msg::Imu> imu_msg;
    std::optional<sensor_msgs::msg::Temperature> temperature;
    std::optional<sensor_msgs::msg::FluidPressure> pressure;
  };

  void start_receive();
  void start_receive_50hz();
  void packet_callback(boost::system::error_code ec, std::size_t bytes_recvd);
  void packet_callback_50hz(boost::system::error_code ec, std::size_t bytes_recvd);
  void parse_eth100_msg(const Eth100HzMsgT &msg);
  void parse_eth50_msg(const Eth50HzESCMsgT &msg);
  static XsensROSMessages parse_xsens_fb(const XsensIMUFrameT &fb_frame);
  void thruster_setpoint_callback(const tauv_msgs::msg::ThrusterSetpoint::SharedPtr msg);
  void sendCallback();

  boost::asio::io_context io_context_;
  udp::socket socket_100_hz_;     // For 100Hz messages
  udp::socket socket_50_hz_;      // For both receiving and sending 50Hz messages
  udp::endpoint remote_endpoint_;
  udp::endpoint remote_endpoint_50hz_;
  udp::endpoint send_endpoint_100hz_;
  std::array<char, 1024> recv_buffer_{};
  std::array<char, 1024> recv_buffer_50hz_{};
  std::thread io_thread_;

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr temperature_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr pressure_publisher_;
  rclcpp::Publisher<tauv_msgs::msg::EscTelemetry>::SharedPtr esc_telemetry_publisher_;
  rclcpp::Publisher<tauv_msgs::msg::DepthSensorFrame>::SharedPtr depth_publisher_;
  rclcpp::Subscription<tauv_msgs::msg::ThrusterSetpoint>::SharedPtr thruster_setpoint_subscriber_;
  rclcpp::TimerBase::SharedPtr send_timer_;
  
  // RPM command data
  std::vector<int16_t> rpms;
  std::vector<bool> enables;
};

#endif //RTVC_H
