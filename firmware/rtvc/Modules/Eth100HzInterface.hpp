/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Interface definitions for Ethernet 100Hz communication
 *
 *****************************************************************************/
 
#pragma once

#include "Eth100HzMessage.hpp"
#include "MTI300Message.hpp"
#include "MS5837Message.hpp"

namespace TAUV {

class Eth100HzInterface {
public:
  Eth100HzInterface(const MTI300Message& mti300_msg, const MS5837Message& ms5837_msg) 
    : mti300_msg_(mti300_msg), ms5837_msg_(ms5837_msg) {}

  const MTI300Message& getMTI300Message() const { return mti300_msg_; }
  const MS5837Message& getMS5837Message() const { return ms5837_msg_; }

private:
  const MTI300Message& mti300_msg_;
  const MS5837Message& ms5837_msg_;
};

}  // namespace TAUV
