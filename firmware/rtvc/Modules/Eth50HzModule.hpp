/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/

#pragma once

#include <cstdint>
#include <string>

#include "Config.hpp"
#include "Eth50HzInterface.hpp"
#include "ModuleBase.hpp"
#include "StaticQueue.hpp"
#include "UdpSocket.hpp"


using std::size_t;

namespace TAUV {

static constexpr size_t max_rx_fb_length = 128;
static constexpr size_t rx_queue_len = 8;

class Eth50HzModule final : public ModuleBase<Eth50HzInterface, Eth50HzMessage> {
 public:
  Eth50HzModule(const Eth50HzInterface& input_interface,
                  Eth50HzMessage& output_msg)
        : ModuleBase(input_interface, output_msg) {}

  ModuleRunResult run() override;
  ModuleInitResult init();

  const char* getName() const override { return "Eth50HzModule"; }
  float getFrequency() const override { return 50.0f; }

 private:
  struct alignas(4) RxBuf {
    std::array<uint8_t, max_rx_fb_length> buf;
    size_t len;
  };

  StaticQueue<RxBuf, rx_queue_len> rx_queue_;

  UdpSocket sock_{};

  void onReceive(const ip_addr_t& addr, uint16_t port, const uint8_t* data, uint16_t len);
};

}  // namespace TAUV
