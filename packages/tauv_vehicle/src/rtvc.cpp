#include "tauv_vehicle/rtvc.h"

RTVCNode::RTVCNode()
    : Node("rtvc_node"),
      // Receive socket, for receiving messages at 100 Hz
      recv_socket_(io_context_, udp::endpoint(udp::v4(), 11003)),
      // Send socket, for sending messages at 50 Hz
      send_socket_(io_context_, udp::endpoint(udp::v4(), 11004)),
      // Sets target address and port
      send_endpoint_(boost::asio::ip::make_address("10.0.0.21"), 11004) {
  imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);
  temperature_publisher_ = this->create_publisher<sensor_msgs::msg::Temperature>("temperature", 10);
  pressure_publisher_ = this->create_publisher<sensor_msgs::msg::FluidPressure>("pressure", 10);
  rpm_command_subscriber_ =
      this->create_subscription<tauv_msgs::msg::RpmCommand>(
          "rpm_command", 10, 
          std::bind(&RTVCNode::rpm_command_callback, this, std::placeholders::_1));

  start_receive();

  std::cout << "Ayo!\n";

  // Spin up io_context in background
  io_thread_ = std::thread([this]() { io_context_.run(); });
}

RTVCNode::~RTVCNode() {
  io_context_.stop();
  if (io_thread_.joinable()) io_thread_.join();
}

void RTVCNode::start_receive() {
  recv_socket_.async_receive_from(
      boost::asio::buffer(recv_buffer_), remote_endpoint_,
      [this](boost::system::error_code ec, std::size_t bytes_recvd) {
        packet_callback(ec, bytes_recvd);
      });
}

void RTVCNode::packet_callback(boost::system::error_code ec, std::size_t bytes_recvd) {
  if (!ec && bytes_recvd > 0) {
    auto fb_root = GetEth100HzMsg(recv_buffer_.data());
    if (!fb_root) {
      RCLCPP_WARN(this->get_logger(),
                  "Failed to parse Eth100HzMsg from buffer");
      start_receive();  // Keep listening
      return;
    }

    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t *>(recv_buffer_.data()), bytes_recvd);
    if (!verifier.VerifyBuffer<Eth100HzMsg>(nullptr)) {
      RCLCPP_WARN(this->get_logger(), "FlatBuffer verification failed");
      start_receive();  // Keep listening
      return;
    }

    Eth100HzMsgT msg;
    fb_root->UnPackTo(&msg);
    parse_eth100_msg(msg);
  }
  start_receive();  // Keep listening
}

void RTVCNode::parse_eth100_msg(const Eth100HzMsgT &msg) {
  // Use a reference instead of copying
  const auto &fb_imu_data = msg.imu_data;

  for (const auto &fb_frame : fb_imu_data) {
    auto msgs = parse_xsens_fb(*fb_frame);
    if (msgs.imu_msg.has_value()) {
      msgs.imu_msg.value().header.stamp = this->get_clock()->now();
      this->imu_publisher_->publish(msgs.imu_msg.value());
    }
    if (msgs.temperature.has_value()) {
      msgs.temperature.value().header.stamp = this->get_clock()->now();
      this->temperature_publisher_->publish(msgs.temperature.value());
    }
    if (msgs.pressure.has_value()) {
      msgs.pressure.value().header.stamp = this->get_clock()->now();
      this->pressure_publisher_->publish(msgs.pressure.value());
    }
  }
}

RTVCNode::XsensROSMessages RTVCNode::parse_xsens_fb(
    const XsensIMUFrameT &fb_frame) {
  XsensROSMessages output_msgs{};

  if (fb_frame.sample_time_fine && fb_frame.orientation &&
      fb_frame.rate_of_turn && fb_frame.free_acceleration) {
    // Orientation
    sensor_msgs::msg::Imu imu_msg;
    imu_msg.orientation.w = fb_frame.orientation->w();
    imu_msg.orientation.x = fb_frame.orientation->x();
    imu_msg.orientation.y = fb_frame.orientation->y();
    imu_msg.orientation.z = fb_frame.orientation->z();

    // Angular velocity (rate_of_turn)
    imu_msg.angular_velocity.x = fb_frame.rate_of_turn->x();
    imu_msg.angular_velocity.y = fb_frame.rate_of_turn->y();
    imu_msg.angular_velocity.z = fb_frame.rate_of_turn->z();

    // Linear acceleration (free_acceleration)
    imu_msg.linear_acceleration.x = fb_frame.free_acceleration->x();
    imu_msg.linear_acceleration.y = fb_frame.free_acceleration->y();
    imu_msg.linear_acceleration.z = fb_frame.free_acceleration->z();

    output_msgs.imu_msg = imu_msg;

    // Leave the header empty (timestamp and frame_id unset)
    // Covariance matrices can be left unset or initialized to identity/zero
    // if needed
    std::fill(std::begin(imu_msg.orientation_covariance),
              std::end(imu_msg.orientation_covariance), 0.0);
    std::fill(std::begin(imu_msg.angular_velocity_covariance),
              std::end(imu_msg.angular_velocity_covariance), 0.0);
    std::fill(std::begin(imu_msg.linear_acceleration_covariance),
              std::end(imu_msg.linear_acceleration_covariance), 0.0);
  }

  if (fb_frame.temperature != 0.0f) {
    sensor_msgs::msg::Temperature temperature_msg{};
    temperature_msg.temperature = fb_frame.temperature;
    output_msgs.temperature = temperature_msg;
  }

  if (fb_frame.pressure != 0.0f) {
    sensor_msgs::msg::FluidPressure pressure_msg{};
    pressure_msg.fluid_pressure = fb_frame.pressure;
    output_msgs.pressure = pressure_msg;
  }

  return output_msgs;
}

void RTVCNode::rpm_command_callback(const tauv_msgs::msg::RpmCommand::SharedPtr msg) {
  // Create the top-level Eth50HzTxMsg
  Eth50HzTxMsgT msg_obj;
  
  auto thruster_command = std::make_unique<ThrusterCommandT>();
  thruster_command->enabled = std::vector<bool>(msg->enables.begin(), msg->enables.end());
  thruster_command->rpm = std::vector<int32_t>(msg->rpms.begin(), msg->rpms.end());
  msg_obj.thruster_command = std::move(thruster_command);

  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(Eth50HzTxMsg::Pack(builder, &msg_obj));
  auto buf = std::make_shared<std::vector<char>>(
    reinterpret_cast<char*>(builder.GetBufferPointer()),
    reinterpret_cast<char*>(builder.GetBufferPointer() + builder.GetSize())
  );

  // Ship it off over UDP (via asio strand
  io_context_.post([this, buf]() {
    boost::system::error_code ec;
    send_socket_.send_to(boost::asio::buffer(*buf), send_endpoint_, 0, ec);
    if (ec)
      RCLCPP_WARN(this->get_logger(), "UDP send error: %s",
                  ec.message().c_str());
  });
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RTVCNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
