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
#include "tauv_msgs/msg/rpm_command.hpp"
#include "tauv_vehicle/generated/eth_msg_jetson_rtvc_50_generated.h"
#include "tauv_vehicle/generated/eth_msg_rtvc_jetson_100_generated.h"

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
  void packet_callback(boost::system::error_code ec, std::size_t bytes_recvd);
  void parse_eth100_msg(const Eth100HzMsgT &msg);
  static XsensROSMessages parse_xsens_fb(const XsensIMUFrameT &fb_frame);
  void rpm_command_callback(const tauv_msgs::msg::RpmCommand::SharedPtr msg);
  void sendCallback();

  boost::asio::io_context io_context_;
  udp::socket recv_socket_;
  udp::socket send_socket_;
  udp::endpoint remote_endpoint_;
  udp::endpoint send_endpoint_;
  std::array<char, 1024> recv_buffer_{};
  std::thread io_thread_;

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr temperature_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::FluidPressure>::SharedPtr pressure_publisher_;
  rclcpp::Subscription<tauv_msgs::msg::RpmCommand>::SharedPtr rpm_command_subscriber_;
  rclcpp::TimerBase::SharedPtr send_timer_;
  
  // RPM command data
  std::vector<int16_t> rpms;
  std::vector<bool> enables;
};

#endif //RTVC_H
