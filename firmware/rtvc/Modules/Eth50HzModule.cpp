/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/

#include "Eth50HzModule.hpp"

#include "eth_msg_jetson_rtvc_generated.h"

using namespace TAUV;

ModuleInitResult Eth50HzModule::init() {
  sock_.bind(Config::Network::jetson_50hz_port);
  sock_.set_receive_callback(
      [this](const ip_addr_t& addr, uint16_t port, const uint8_t* data,
             uint16_t len) { this->onReceive(addr, port, data, len); });

  return ModuleInitResult::OK;
}

ModuleRunResult Eth50HzModule::run() {

  RxBuf buf;

  size_t received_counter = 0;
  while (rx_queue_.receive(buf, 0)) {
    const auto& [arr, size] = buf;
    flatbuffers::Verifier verifier(arr.begin(), size);
    bool buffer_valid = TAUV_FB::VerifyEth50HzTxMsgBuffer(verifier);
    if (!buffer_valid) {
      // todo: warn
      continue;
    }
    const TAUV_FB::Eth50HzTxMsg *eth_msg = TAUV_FB::GetEth50HzTxMsg(arr.begin());
    const TAUV_FB::ThrusterCommand *tc = eth_msg->thruster_command();
    assert (tc->enabled()->size() == output_msg_.esc_enable.size());
    for (size_t i = 0; i < tc->enabled()->size(); ++i) {
      output_msg_.esc_enable[i] = static_cast<bool>(tc->enabled()->Get(i));
      output_msg_.esc_rpm[i] = tc->rpm()->Get(i);
    }

    ++received_counter;
  }

  if (received_counter < 1) {
    return ModuleRunResult::OUTPUT_INVALID;
  }

  if (received_counter > 2) {
    // todo: send warning
  }

  return ModuleRunResult::OK;
}

void Eth50HzModule::onReceive(const ip_addr_t& addr, uint16_t port,
                              const uint8_t* data, uint16_t len) {
  if (len > max_rx_fb_length) {
    // todo: handle error
    return;
  }
  std::array<uint8_t, max_rx_fb_length> fb_data{};
  std::copy_n(data, len, fb_data.begin());
  rx_queue_.send(RxBuf{fb_data, len});
}
