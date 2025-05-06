#include <array>
#include <boost/asio.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "flatbuffers/flatbuffers.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "tauv_vehicle/generated/eth_msg_rtvc_jetson_generated.h"
#include "tauv_vehicle/generated/eth_msg_jetson_rtvc_generated.h"

#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "tauv_msgs/msg/rpm_command.hpp"

using boost::asio::ip::udp;
using namespace TAUV_FB;

class UdpCommNode : public rclcpp::Node {
public:
  UdpCommNode()
      : Node("udp_comm_node"),
        // Receive socket, for receiving messages at 100 Hz
        recv_socket_(io_context_, udp::endpoint(udp::v4(), 11003)),
        // Send socket, for sending messages at 50 Hz
        send_socket_(io_context_, udp::endpoint(udp::v4(), 11004)),
        // Sets target address and port
        send_endpoint_(boost::asio::ip::make_address("10.0.0.21"), 11004) {

    imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);

    rpm_command_subscriber_ = this->create_subscription<tauv_msgs::msg::RpmCommand>(
                                "rpm_command", 10, std::bind(&UdpCommNode::rpm_command_callback, this, std::placeholders::_1)); 

    start_receive();

    std::cout << "Ayo!\n";

    // Run a 50 Hz timer for sending messages
    send_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),
      std::bind(&UdpCommNode::sendCallback, this)
    );

    // Spin up io_context in background
    io_thread_ = std::thread([this]() { io_context_.run(); });
  }

  ~UdpCommNode() {
    io_context_.stop();
    if (io_thread_.joinable())
      io_thread_.join();
  }

private:

  // rpm and enable arrays
  int32_t rpms[8]    = {0, 0, 0, 0, 0, 0, 0, 0};
  uint8_t enables[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  void start_receive() {
    recv_socket_.async_receive_from(
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

  void rpm_command_callback(const tauv_msgs::msg::RpmCommand::SharedPtr msg) {
    for (size_t i = 0; i < msg->rpms.size; ++i) {
      rpms[i] = msg->rpms[i];
      enables[i] = msg->enables[i];
    }
  }

  // Sends ESC commands
  void sendCallback() {

    // Create the top-level Eth50HzTxMsg
    Eth50HzTxMsgT msg_obj;
    msg_obj.thruster_command = std::make_unique<RpmCommand>(rpms, enables);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Eth50HzTxMsg::Pack(builder, &msg_obj));
    // copy payload bytes into a heap‐allocated vector
    auto buf = std::make_shared<std::vector<char>>(
      reinterpret_cast<char*>(builder.GetBufferPointer()),
      reinterpret_cast<char*>(builder.GetBufferPointer() + builder.GetSize())
    );
    std::cout << "this is def new shit\n";


    // Ship it off over UDP (via asio strand
    io_context_.post([this, buf]()
    {
        boost::system::error_code ec;
        send_socket_.send_to(boost::asio::buffer(*buf), send_endpoint_, 0, ec);
        std::cout << "i'm sending stuff\n";
        if (ec)
          RCLCPP_WARN(this->get_logger(), "UDP send error: %s", ec.message().c_str());
    });
  }


  boost::asio::io_context io_context_;
  udp::socket recv_socket_;
  udp::socket send_socket_;
  udp::endpoint remote_endpoint_;
  udp::endpoint send_endpoint_;
  std::array<char, 1024> recv_buffer_;
  std::thread io_thread_;

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  rclcpp::Subscription<tauv_msgs::msg::RpmCommand>::SharedPtr rpm_command_subscriber_;
  rclcpp::TimerBase::SharedPtr  send_timer_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<UdpCommNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
