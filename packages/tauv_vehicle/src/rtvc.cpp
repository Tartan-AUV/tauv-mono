#include <chrono>
#include <memory>
#include <thread>
#include <functional>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "rtvc_generated.h"

using namespace std::chrono_literals;

#define UDP_PORT 11003
#define BUFFER_SIZE 4096

class RTVCNode : public rclcpp::Node
{
public:
  RTVCNode() 
  : Node("udp_flatbuffer_node"), socket_fd_(-1)
  {
    // Parameters
    this->declare_parameter("udp_port", UDP_PORT);
    int port = this->get_parameter("udp_port").as_int();
    
    // Publisher for parsed data
    publisher_ = this->create_publisher<std_msgs::msg::string>("parsed_flatbuffer", 10);

    // Initialize socket and start listening in a separate thread
    if (init_socket(port)) {
      RCLCPP_INFO(this->get_logger(), "UDP socket initialized on port %d", port);
      listen_thread_ = std::thread(&RTVCNode::listen_for_data, this);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize UDP socket");
    }
  }

  ~RTVCNode()
  {
    // Stop the thread and close the socket
    if (socket_fd_ >= 0) {
      should_exit_ = true;
      if (listen_thread_.joinable()) {
        listen_thread_.join();
      }
      close(socket_fd_);
      RCLCPP_INFO(this->get_logger(), "UDP socket closed");
    }
  }

private:
  bool init_socket(int port)
  {
    // Create UDP socket
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      RCLCPP_ERROR(this->get_logger(), "Socket creation failed");
      return false;
    }

    // Clear and set up server address structure
    memset(&server_addr_, 0, sizeof(server_addr_));
    server_addr_.sin_family = AF_INET;
    server_addr_.sin_addr.s_addr = INADDR_ANY;
    server_addr_.sin_port = htons(port);

    // Bind socket to address and port
    if (bind(socket_fd_, (const struct sockaddr *)&server_addr_, sizeof(server_addr_)) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Bind failed");
      close(socket_fd_);
      socket_fd_ = -1;
      return false;
    }

    // Set socket to non-blocking mode
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);

    return true;
  }

  void listen_for_data()
  {
    char buffer[BUFFER_SIZE];
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    RCLCPP_INFO(this->get_logger(), "Starting UDP listener thread");

    while (rclcpp::ok() && !should_exit_) {
      memset(buffer, 0, BUFFER_SIZE);
      
      // Receive data (non-blocking)
      int len = recvfrom(socket_fd_, buffer, BUFFER_SIZE, 0, 
                        (struct sockaddr *)&client_addr, &client_len);
      
      if (len > 0) {
        // Got data, try to parse as FlatBuffer
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
        
        RCLCPP_INFO(this->get_logger(), 
                   "Received %d bytes from %s:%d", 
                   len, 
                   client_ip, 
                   ntohs(client_addr.sin_port));
        
        parse_flatbuffer(buffer, len);
      } 
      else if (len < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        // Real error occurred
        RCLCPP_ERROR(this->get_logger(), "recvfrom error: %s", strerror(errno));
      }
      
      // Small sleep to prevent CPU hogging
      std::this_thread::sleep_for(10ms);
    }
  }

  void parse_flatbuffer(const char* data, size_t len)
  {
    // Verify the buffer and get root
    // This is where you'd implement your FlatBuffer parsing logic
    try {
      // Example parsing code (replace with your actual schema):
      /*
      // Verify the buffer
      flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(data), len);
      if (!YourRootType::VerifyBuffer(verifier)) {
        RCLCPP_WARN(this->get_logger(), "Invalid FlatBuffer format");
        return;
      }

      // Get the root of the FlatBuffer
      auto message = GetYourRootType(data);
      
      // Extract data from the message
      std::string some_field = message->some_field()->str();
      int some_value = message->some_value();
      
      // Publish to ROS2 topic
      auto msg = std::make_unique<std_msgs::msg::String>();
      msg->data = "Parsed data: " + some_field + ", " + std::to_string(some_value);
      publisher_->publish(*msg);
      */
      
      // For now, just publish that we received something
      auto msg = std::make_unique<std_msgs::msg::String>();
      msg->data = "Received flatbuffer data of size " + std::to_string(len);
      publisher_->publish(*msg);
      
      RCLCPP_INFO(this->get_logger(), "Successfully parsed FlatBuffer message");
    }
    catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Error parsing FlatBuffer: %s", e.what());
    }
  }

  int socket_fd_;
  struct sockaddr_in server_addr_;
  std::thread listen_thread_;
  bool should_exit_ = false;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RTVCNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}