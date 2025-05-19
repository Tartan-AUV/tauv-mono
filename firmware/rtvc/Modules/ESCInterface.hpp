/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Interface for ESC module with telemetry support
 *
 *****************************************************************************/
 
#pragma once

#include <array>
#include <cstdint>

#include "Config.hpp"
#include "Eth50HzMessage.hpp"
#include "VESC_UART.hpp"

namespace TAUV {

class ESCInterface {
public:
  ESCInterface(const Eth50HzMessage &eth_50hz_msg) : eth_50hz_msg(eth_50hz_msg) {}

  const std::array<int32_t, Config::Thrusters::number_escs>& get_rpm() const {
    return eth_50hz_msg.esc_rpm;
  }

  const std::array<bool, Config::Thrusters::number_escs>& get_enable() const {
    return eth_50hz_msg.esc_enable;
  }

  bool is_valid() const {
    return eth_50hz_msg.valid;
  }

private:
  // Messages
  const Eth50HzMessage &eth_50hz_msg;
};

}  // namespace TAUV

