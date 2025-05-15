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

#include <cstddef>

#include "ESCModule.hpp"
#include "Eth50HzInterface.hpp"
#include "Eth50HzModule.hpp"
#include "IntervalTask.hpp"
#include "stm32f7xx_hal.h"

using std::size_t;

namespace TAUV {

class Task50Hz final : public IntervalTask {
 public:
  struct Resources {
    std::array<UART_HandleTypeDef*, Config::Thrusters::esc_groups.size()> uarts;
  };

  bool init(std::unique_ptr<Resources> resources);

 private:
  void run() override;

  // Message
  Eth50HzMessage eth_50hz_msg_{};
  ESCMessage esc_msg_{};

  // Interfaces
  Eth50HzInterface eth_50hz_interface{};
  ESCInterface esc_interface_{eth_50hz_msg_};

  // Modules
  Eth50HzModule eth_50hz_module{eth_50hz_interface, eth_50hz_msg_};
  ESCModule esc_module{esc_interface_, esc_msg_, };

  // Resources
  std::unique_ptr<Resources> resources_ = nullptr;
};

}
