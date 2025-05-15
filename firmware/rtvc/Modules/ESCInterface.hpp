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
#include "Eth50HzInterface.hpp"

namespace TAUV {

struct ESCMessage {

};

class ESCInterface {
public:
  ESCInterface(const Eth50HzMessage &eth_50hz_msg) : eth_50hz_msg(eth_50hz_msg) {}

  const std::array<int32_t, Config::Thrusters::number_escs>& get_rpm() const {
    return eth_50hz_msg.esc_rpm;
  }

  const std::array<bool, Config::Thrusters::number_escs>& get_enable() const {
    return eth_50hz_msg.esc_enable;
  }

private:
  // Messages
  const Eth50HzMessage &eth_50hz_msg;
};

}

