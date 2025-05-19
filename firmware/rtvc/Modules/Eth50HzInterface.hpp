/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Interface for 50Hz Ethernet communication with telemetry support
 *
 *****************************************************************************/
 
#pragma once

#include "ESCMessage.hpp"
#include "Eth50HzMessage.hpp"

namespace TAUV {

class Eth50HzInterface {
public:
  Eth50HzInterface(const ESCMessage& esc_msg) : esc_msg_(esc_msg) {}

  // Get access to the ESC telemetry data
  const ESCMessage& getESCMessage() const { return esc_msg_; }

private:
  const ESCMessage& esc_msg_;
};

}  // namespace TAUV

