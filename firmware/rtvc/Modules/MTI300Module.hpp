/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      MTI300 IMU Module for processing inertial data from XSens MTI-300 IMU
 *      using interrupt-driven UART communication
 *
 *****************************************************************************/

#pragma once

#include "MTI300Interface.hpp"
#include "MTI300Message.hpp"
#include "ModuleBase.hpp"
#include "Mti300.hpp"
#include "stm32f7xx_hal.h"

namespace TAUV {

class MTI300Module : public ModuleBase<MTI300InputInterface, MTI300Message> {
 public:
  MTI300Module(const MTI300InputInterface &input_interface,
               MTI300Message &output_msg)
      : ModuleBase<MTI300InputInterface, MTI300Message>(input_interface,
                                                        output_msg) {}

  const char *getName() const override { return "MTI300"; }
  float getFrequency() const override { return 100.0f; }

  ModuleInitResult init(UART_HandleTypeDef *uart);
  ModuleRunResult run() override;

 private:
  MTI300 imu_driver_;
  UART_HandleTypeDef *uart_ = nullptr;
};

}  // namespace TAUV
