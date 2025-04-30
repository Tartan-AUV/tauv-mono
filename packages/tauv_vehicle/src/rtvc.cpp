#include <array>
#include <boost/asio.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "flatbuffers/flatbuffers.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "tauv_vehicle/generated/eth_msg_rtvc_jetson_generated.h"

#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "sensor_msgs/msg/imu.hpp"

using boost::asio::ip::udp;
using namespace TAUV_FB;

class UdpListenerNode : public rclcpp::Node {
public:
  UdpListenerNode()
      : Node("udp_listener_node"),
        socket_(io_context_, udp::endpoint(udp::v4(), 11003)) {
    imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);

    start_receive();

    std::cout << "Ayo!\n";

    // Spin up io_context in background
    io_thread_ = std::thread([this]() { io_context_.run(); });
  }

  ~UdpListenerNode() {
    io_context_.stop();
    if (io_thread_.joinable())
      io_thread_.join();
  }

private:
  void start_receive() {
    socket_.async_receive_from(
        boost::asio::buffer(recv_buffer_), remote_endpoint_,
        [this](boost::system::error_code ec, std::size_t bytes_recvd) {
          packet_callback(ec, bytes_recvd);
        });
  }

  void packet_callback(boost::system::error_code ec, std::size_t bytes_recvd) {
    if (!ec && bytes_recvd > 0) {

      auto fb_root = GetEth100HzMsg(recv_buffer_.data());
      if (!fb_root) {
        RCLCPP_WARN(this->get_logger(),
                    "Failed to parse Eth100HzMsg from buffer");
        start_receive(); // Keep listening
        return;
      }

      flatbuffers::Verifier verifier(
          reinterpret_cast<const uint8_t *>(recv_buffer_.data()), bytes_recvd);
      if (!verifier.VerifyBuffer<Eth100HzMsg>(nullptr)) {
        RCLCPP_WARN(this->get_logger(), "FlatBuffer verification failed");
        start_receive(); // Keep listening
        return;
      }

      Eth100HzMsgT msg;
      fb_root->UnPackTo(&msg);
      parse_eth100_msg(msg);
    }
    start_receive(); // Keep listening
  }

  void parse_eth100_msg(const Eth100HzMsgT &msg) {
    auto fb_imu_data = msg.imu_data;

    for (const auto &fb_frame : fb_imu_data) {
      sensor_msgs::msg::Imu imu_msg = imu_msg_from_xsens_flatbuf(fb_frame);
      imu_msg.header.stamp = this->get_clock()->now();
      this->imu_publisher_->publish(imu_msg);
    }
  }

  sensor_msgs::msg::Imu
  imu_msg_from_xsens_flatbuf(const XsensIMUFrame &fb_frame) {
    sensor_msgs::msg::Imu imu_msg;

    // Orientation
    imu_msg.orientation.w = fb_frame.orientation().w();
    imu_msg.orientation.x = fb_frame.orientation().x();
    imu_msg.orientation.y = fb_frame.orientation().y();
    imu_msg.orientation.z = fb_frame.orientation().z();

    // Angular velocity (rate_of_turn)
    imu_msg.angular_velocity.x = fb_frame.rate_of_turn().x();
    imu_msg.angular_velocity.y = fb_frame.rate_of_turn().y();
    imu_msg.angular_velocity.z = fb_frame.rate_of_turn().z();

    // Linear acceleration (free_acceleration)
    imu_msg.linear_acceleration.x = fb_frame.free_acceleration().x();
    imu_msg.linear_acceleration.y = fb_frame.free_acceleration().y();
    imu_msg.linear_acceleration.z = fb_frame.free_acceleration().z();

    // Leave the header empty (timestamp and frame_id unset)
    // Covariance matrices can be left unset or initialized to identity/zero if
    // needed
    std::fill(std::begin(imu_msg.orientation_covariance),
              std::end(imu_msg.orientation_covariance), 0.0);
    std::fill(std::begin(imu_msg.angular_velocity_covariance),
              std::end(imu_msg.angular_velocity_covariance), 0.0);
    std::fill(std::begin(imu_msg.linear_acceleration_covariance),
              std::end(imu_msg.linear_acceleration_covariance), 0.0);

    return imu_msg;
  }

  boost::asio::io_context io_context_;
  udp::socket socket_;
  udp::endpoint remote_endpoint_;
  std::array<char, 1024> recv_buffer_;
  std::thread io_thread_;

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<UdpListenerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
