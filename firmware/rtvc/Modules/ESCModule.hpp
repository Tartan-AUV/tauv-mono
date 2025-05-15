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
#include <ranges>

#include "ESCInterface.hpp"
#include "ModuleBase.hpp"
#include "VESC_UART.hpp"

namespace TAUV {

class ESCModule : public ModuleBase<ESCInterface, ESCMessage> {
 public:
  ESCModule(const ESCInterface &input_interface, ESCMessage &output_msg)
      : ModuleBase<ESCInterface, ESCMessage>(input_interface, output_msg) {}

  const char *getName() const override { return "ESC"; }
  float getFrequency() const override { return 50.0f; }

  ModuleInitResult init(const std::array<UART_HandleTypeDef *, Config::Thrusters::num_groups> &uarts);

  ModuleRunResult run() override;

 private:
  std::array<UART_HandleTypeDef *, Config::Thrusters::num_groups> uarts{};
  std::array<VESC::VESC_UART, Config::Thrusters::num_groups> drivers{};
};

}  // namespace TAUV
