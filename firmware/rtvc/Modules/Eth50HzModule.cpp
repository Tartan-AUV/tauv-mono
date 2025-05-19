/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Ethernet module for 50Hz communication with ESC telemetry support
 *
 *****************************************************************************/

#include "Eth50HzModule.hpp"

#include "Logging.hpp"
#include "eth_msg_jetson_rtvc_50_generated.h"
#include "eth_msg_rtvc_jetson_50_generated.h"
#include "lwip/inet.h"

// External IP address of Jetson
extern ip_addr_t jetsonAddr;

using namespace TAUV;

ModuleInitResult Eth50HzModule::init() {
  sock_.init();
  sock_.bind(Config::Network::jetson_50hz_port);
  sock_.set_receive_callback(
      [this](const ip_addr_t& addr, uint16_t port, const uint8_t* data,
             uint16_t len) { this->onReceive(addr, port, data, len); });

  LOG_INFO("Eth50HzModule: Initialized UDP socket on port %d", Config::Network::jetson_50hz_port);
  return ModuleInitResult::OK;
}

ModuleRunResult Eth50HzModule::run() {
  RxBuf buf;

  size_t received_counter = 0;
  // Set message as invalid initially
  output_msg_.valid = false;
  
  // Process received commands from Jetson
  while (rx_queue_.receive(buf, 0)) {
    const auto& [arr, size] = buf;
    flatbuffers::Verifier verifier(arr.begin(), size);
    bool buffer_valid = TAUV_FB::VerifyEth50HzTxMsgBuffer(verifier);
    if (!buffer_valid) {
      LOG_WARN("Eth50HzModule: Received invalid FlatBuffer message");
      continue;
    }
    const TAUV_FB::Eth50HzTxMsg *eth_msg = TAUV_FB::GetEth50HzTxMsg(arr.begin());
    const TAUV_FB::ThrusterCommand *tc = eth_msg->thruster_command();
    assert (tc->enabled()->size() == output_msg_.esc_enable.size());
    for (size_t i = 0; i < output_msg_.esc_enable.size(); ++i) {
      output_msg_.esc_enable[i] = static_cast<bool>(tc->enabled()->Get(i));
      output_msg_.esc_rpm[i] = tc->rpm()->Get(i);
    }

    // Mark message as valid since we processed a valid message
    output_msg_.valid = true;
    ++received_counter;
  }

  if (received_counter > 2) {
    LOG_WARN("Eth50HzModule: Received %d messages in one cycle (expected 1, at most 2)", received_counter);
  }

  // Send ESC telemetry to Jetson
  if (input_interface_.getESCMessage().telemetry.size() > 0) {
    // Create FlatBuffer for ESC telemetry
    flatbuffers::FlatBufferBuilder fbb(max_tx_fb_length);

    // Build ESCFrame objects for each ESC
    std::vector<flatbuffers::Offset<TAUV_FB::ESCFrame>> esc_frames;

    for (size_t i = 0; i < Config::Thrusters::number_escs; i++) {
      const auto& telemetry = input_interface_.getESCMessage().telemetry[i];

      // Only include valid telemetry data
      if (telemetry.data_valid) {
        auto frame = TAUV_FB::CreateESCFrame(
          fbb,
          static_cast<uint8_t>(i),              // ESC ID
          telemetry.rpm,                        // RPM
          telemetry.voltage,                    // Voltage
          telemetry.current,                    // Current
          telemetry.temperature_motor,          // Temperature (motor)
          static_cast<uint8_t>(telemetry.fault_code)  // Fault code
        );

        esc_frames.push_back(frame);
      }
    }

    // Create vector of ESC frames
    auto esc_frames_vector = fbb.CreateVector(esc_frames);

    // Create root message
    auto eth_msg = TAUV_FB::CreateEth50HzESCMsg(fbb, esc_frames_vector);

    // Finish buffer
    fbb.Finish(eth_msg);

    // Send telemetry to Jetson
    sock_.send(jetsonAddr, Config::Network::jetson_50hz_port,
               fbb.GetBufferPointer(), fbb.GetSize());
  }

  return ModuleRunResult::OK;
}

void Eth50HzModule::onReceive(const ip_addr_t& addr, uint16_t port,
                              const uint8_t* data, uint16_t len) {
  if (len > max_rx_fb_length) {
    LOG_WARN("Eth50HzModule: Received message too large (%d bytes, max %d)",
             len, max_rx_fb_length);
    return;
  }
  std::array<uint8_t, max_rx_fb_length> fb_data{};
  std::copy_n(data, len, fb_data.begin());
  rx_queue_.send(RxBuf{fb_data, len});
}
