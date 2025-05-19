/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Message definitions for Ethernet 100Hz communication
 *
 *****************************************************************************/
 
#pragma once

#include "MTI300Message.hpp"

namespace TAUV {

struct Eth100HzMessage {
  // Reference to the MTI300 message data
  const MTI300Message* mti300_msg = nullptr;
};

}  // namespace TAUV
