/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Interface and message definitions for Ethernet 100Hz communication
 *
 *****************************************************************************/
 
#pragma once

#include <array>
#include <cstdint>

#include "MTI300Interface.hpp"

using std::size_t;

namespace TAUV {

struct Eth100HzMessage {
  // Reference to the MTI300 message data
  const MTI300Message* mti300_msg = nullptr;
};

class Eth100HzInterface {
public:
  Eth100HzInterface(const MTI300Message& mti300_msg) : mti300_msg_(mti300_msg) {}

  const MTI300Message& getMTI300Message() const { return mti300_msg_; }

private:
  const MTI300Message& mti300_msg_;
};

}
