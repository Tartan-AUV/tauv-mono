#include <rclcpp/rclcpp.hpp>
#include <boost/asio.hpp>
#include <memory>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "tauv_vehicle/generated/rtvc_generated.h"

#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/quaternion.hpp"

using boost::asio::ip::udp;

class UdpListenerNode : public rclcpp::Node {
public:
    UdpListenerNode()
        : Node("udp_listener_node"),
          socket_(io_context_, udp::endpoint(udp::v4(), 11003)) 
    {
        imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);

        start_receive();

        // Spin up io_context in background
        io_thread_ = std::thread([this]() { io_context_.run(); });
    }

    ~UdpListenerNode() {
        io_context_.stop();
        if (io_thread_.joinable()) io_thread_.join();
    }

private:
    void start_receive() {
        socket_.async_receive_from(
            boost::asio::buffer(recv_buffer_), remote_endpoint_,
            [this](boost::system::error_code ec, std::size_t bytes_recvd) {
            });
    }

    void packet_callback(boost::system::error_code ec, std::size_t bytes_recvd) {
        if (!ec && bytes_recvd > 0) {
            TAUV::Eth100HzMsgT eth_100hz_msg;
            TAUV::GetEth100HzMsg(recv_buffer_.data())->UnPackTo(&eth_100hz_msg);
            parse_eth100_msg(&eth_100hz_msg);
        }
        start_receive(); // Keep listening
    }

    sensor_msgs::msg::Imu imu_msg_from_xsens_flatbuf(const TAUV::XsensIMUFrame* fb_frame) {
        sensor_msgs::msg::Imu imu_msg;

        // Orientation
        imu_msg.orientation.w = fb_frame->orientation().w();
        imu_msg.orientation.x = fb_frame->orientation().x();
        imu_msg.orientation.y = fb_frame->orientation().y();
        imu_msg.orientation.z = fb_frame->orientation().z();

        // Angular velocity (rate_of_turn)
        imu_msg.angular_velocity.x = fb_frame->rate_of_turn().x();
        imu_msg.angular_velocity.y = fb_frame->rate_of_turn().y();
        imu_msg.angular_velocity.z = fb_frame->rate_of_turn().z();

        // Linear acceleration (free_acceleration)
        imu_msg.linear_acceleration.x = fb_frame->free_acceleration().x();
        imu_msg.linear_acceleration.y = fb_frame->free_acceleration().y();
        imu_msg.linear_acceleration.z = fb_frame->free_acceleration().z();

        // Leave the header empty (timestamp and frame_id unset)
        // Covariance matrices can be left unset or initialized to identity/zero if needed
        std::fill(std::begin(imu_msg.orientation_covariance), std::end(imu_msg.orientation_covariance), 0.0);
        std::fill(std::begin(imu_msg.angular_velocity_covariance), std::end(imu_msg.angular_velocity_covariance), 0.0);
        std::fill(std::begin(imu_msg.linear_acceleration_covariance), std::end(imu_msg.linear_acceleration_covariance), 0.0);

        return imu_msg;
    }

    void parse_eth100_msg(const TAUV::Eth100HzMsg *msg) {
        auto fb_imu_data = msg->imu_data();
    
        if (!fb_imu_data) return result;

        for (auto fb_frame : *fb_imu_data) {
            sensor_msgs::msg::Imu imu_msg = imu_msg_from_xsens_flatbuf(fb_frame);
            imu_msg.header.stamp = this->get_clock()->now();
            this->imu_publisher_->publish(imu_msg);
        }
    }

    boost::asio::io_context io_context_;
    udp::socket socket_;
    udp::endpoint remote_endpoint_;
    std::array<char, 1024> recv_buffer_;
    std::thread io_thread_;

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UdpListenerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
