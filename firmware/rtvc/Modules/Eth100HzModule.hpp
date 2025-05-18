/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Ethernet module for sending 100Hz IMU data to the Jetson
 *
 *****************************************************************************/

#pragma once

#include <cstdint>
#include <string>
#include <array>

#include "Config.hpp"
#include "Eth100HzInterface.hpp"
#include "ModuleBase.hpp"
#include "StaticQueue.hpp"
#include "UdpSocket.hpp"

using std::size_t;

namespace TAUV {

// Define max sizes for buffers
static constexpr size_t max_tx_fb_length = 1024;

class Eth100HzModule final : public ModuleBase<Eth100HzInterface, Eth100HzMessage> {
 public:
  Eth100HzModule(const Eth100HzInterface& input_interface,
                  Eth100HzMessage& output_msg)
        : ModuleBase(input_interface, output_msg) {}

  ModuleRunResult run() override;
  ModuleInitResult init();

  const char* getName() const override { return "Eth100HzModule"; }
  float getFrequency() const override { return 100.0f; }

 private:
  UdpSocket sock_{};
  std::array<uint8_t, max_tx_fb_length> fb_builder_buffer_;
};

}  // namespace TAUV
