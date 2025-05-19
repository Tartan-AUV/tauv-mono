/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      MS5837 Pressure/Temperature Sensor Module for processing 
 *      depth/temperature data using I2C communication
 *
 *****************************************************************************/

#pragma once

#include "MS5837Interface.hpp"
#include "MS5837Message.hpp"
#include "ModuleBase.hpp"
#include "MS5837.h"
#include "stm32f7xx_hal.h"

namespace TAUV {

class MS5837Module : public ModuleBase<MS5837InputInterface, MS5837Message> {
 public:
  MS5837Module(const MS5837InputInterface &input_interface,
               MS5837Message &output_msg)
      : ModuleBase<MS5837InputInterface, MS5837Message>(input_interface,
                                                        output_msg) {}

  const char *getName() const override { return "MS5837"; }
  float getFrequency() const override { return 100.0f; }

  ModuleInitResult init(I2C_HandleTypeDef *hi2c);
  ModuleRunResult run() override;

 private:
  MS5837 sensor_driver_;
  I2C_HandleTypeDef *hi2c_ = nullptr;
};

}  // namespace TAUV
