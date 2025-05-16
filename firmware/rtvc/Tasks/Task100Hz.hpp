/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      Task running at 100Hz, including MTI300 IMU processing
 *
 *****************************************************************************/
 
#pragma once

#include <cstddef>
#include <memory>

#include "IntervalTask.hpp"
#include "MTI300Module.hpp"
#include "MTI300Interface.hpp"
#include "stm32f7xx_hal.h"

using std::size_t;

namespace TAUV {

class Task100Hz final : public IntervalTask {
 public:
  struct Resources {
    // UART handle for the MTI300 IMU
    UART_HandleTypeDef* imu_uart = nullptr;
  };

  bool init(std::unique_ptr<Resources> resources);

 private:
  void run() override;

  // Messages
  MTI300Message mti300_msg_{};

  // Interfaces
  MTI300InputInterface mti300_input_interface_{};

  // Modules
  MTI300Module mti300_module_{mti300_input_interface_, mti300_msg_};

  // Resources
  std::unique_ptr<Resources> resources_ = nullptr;
};

}
