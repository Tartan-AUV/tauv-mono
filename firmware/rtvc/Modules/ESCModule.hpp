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

  ModuleInitResult init(UART_HandleTypeDef *left_uart, UART_HandleTypeDef *right_uart);

  ModuleRunResult run() override;

 private:
  VESC::VESC_UART driver{};
  UART_HandleTypeDef *left_uart;
  UART_HandleTypeDef *right_uart;
};

}  // namespace TAUV
