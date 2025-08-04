/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      Task running at 100Hz, including MTI300 IMU processing,
 *      MS5837 pressure/depth sensor, and 100Hz ethernet communication
 *
 *****************************************************************************/
 
#pragma once

#include <cstddef>
#include <memory>

#include "Eth100HzInterface.hpp"
#include "Eth100HzModule.hpp"
#include "IntervalTask.hpp"
#include "MS5837Interface.hpp"
#include "MS5837Module.hpp"
#include "MTI300Interface.hpp"
#include "MTI300Module.hpp"
#include "stm32f7xx_hal.h"

using std::size_t;

namespace TAUV {

class Task100Hz final : public IntervalTask {
 public:
  struct Resources {
    UART_HandleTypeDef* imu_uart = nullptr;
    I2C_HandleTypeDef* depth_i2c = nullptr;
  };

  bool init(std::unique_ptr<Resources> resources);

 private:
  void run() override;

  // Output messages
  MTI300Message mti300_output_msg_{};
  MS5837Message ms5837_output_msg_{};
  Eth100HzMessage eth_output_msg_{};

  // Input interfaces
  MTI300InputInterface mti300_input_{};
  MS5837InputInterface ms5837_input_{};
  Eth100HzInterface eth_input_{mti300_output_msg_, ms5837_output_msg_};

  // Modules
  MTI300Module mti300_module_{mti300_input_, mti300_output_msg_};
  MS5837Module ms5837_module_{ms5837_input_, ms5837_output_msg_};
  Eth100HzModule eth_module_{eth_input_, eth_output_msg_};

  // Resources
  std::unique_ptr<Resources> resources_ = nullptr;
};

}
