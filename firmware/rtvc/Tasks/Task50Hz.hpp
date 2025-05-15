/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/
 
#pragma once
#include "Eth50HzModule.hpp"
#include "Eth50HzInterface.hpp"
#include "IntervalTask.hpp"

#include <cstddef>

using std::size_t;

namespace TAUV {

class Task50Hz final : public IntervalTask {

 public:
  bool init();

 private:
  void run() override;

  // Message
  Eth50HzMessage eth_50hz_msg;

  // Interfaces
  Eth50HzInterface eth_50hz_interface;

  // Modules
  Eth50HzModule eth_50hz_module{eth_50hz_interface, eth_50hz_msg};
};

}
