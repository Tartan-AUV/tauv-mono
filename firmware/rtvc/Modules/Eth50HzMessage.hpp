/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Message definitions for 50Hz Ethernet communication with telemetry support
 *
 *****************************************************************************/
 
#pragma once

#include <array>
#include <cstdint>

#include "Config.hpp"

namespace TAUV {

struct Eth50HzMessage {
  bool valid = false;  // Indicates if the message contains valid data
  std::array<int32_t, Config::Thrusters::number_escs> esc_rpm;
  std::array<bool, Config::Thrusters::number_escs> esc_enable;
};

}  // namespace TAUV
