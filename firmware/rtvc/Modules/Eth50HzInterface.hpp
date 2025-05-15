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
 
#pragma once

#include <array>
#include <cstdint>

using std::size_t;

#include "Config.hpp"

namespace TAUV {

struct Eth50HzMessage {

  std::array<int32_t, Config::Thrusters::number_escs> esc_rpm;
  std::array<bool, Config::Thrusters::number_escs> esc_enable;

};

class Eth50HzInterface {
public:
  Eth50HzInterface() = default;
};

}

