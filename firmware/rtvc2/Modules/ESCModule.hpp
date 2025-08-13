/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      ESC module header with telemetry collection support
 *
 *****************************************************************************/

#pragma once

#include <array>
#include <cstdint>

#include "ESCInterface.hpp"
#include "ESCMessage.hpp"
#include "ModuleBase.hpp"
#include "VESC_UART.hpp"

namespace TAUV {

class ESCModule final : public ModuleBase<ESCInterface, ESCMessage> {
 public:
  ESCModule(const ESCInterface &input_interface, ESCMessage &output_msg)
      : ModuleBase<ESCInterface, ESCMessage>(input_interface, output_msg) {}

  const char *getName() const override { return "ESC"; }
  float getFrequency() const override { return 50.0f; }

  ModuleInitResult init(const std::array<UART_HandleTypeDef *, Config::Thrusters::num_groups> &uarts);

  ModuleRunResult run() override;

 private:
  // VESC UART interfaces for the ESC groups
  std::array<VESC::VESC_UART, Config::Thrusters::num_groups> vesc_interfaces_;
  std::array<UART_HandleTypeDef *, Config::Thrusters::num_groups> uarts{};
  
  // Killswitch latch state - once triggered, stays active until system reset
  bool killswitch_latched_ = false;
};

}  // namespace TAUV
