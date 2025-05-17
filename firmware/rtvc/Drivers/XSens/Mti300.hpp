/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/
 
#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "RingBuffer.hpp"
#include "StaticQueue.hpp"
#include "stm32f7xx_hal.h"

namespace TAUV {

class MTI300 {
public:
  struct MTData2Message {
    // Quaternion [w, x, y, z]
    std::optional<std::array<float, 4>> quaternion;
    
    // Free acceleration [ax, ay, az] in m/s²
    std::optional<std::array<float, 3>> freeAcceleration;
    
    // Angular velocity [gx, gy, gz] in rad/s
    std::optional<std::array<float, 3>> angularVelocity;
    
    // Sample timestamp in nanoseconds
    std::optional<uint32_t> sampleTimeFine;
  };

  MTI300();

  void init(UART_HandleTypeDef *uart);

  // to be called from ISR
  static void uartRxCallback(size_t len);

  size_t processQueuedMessages(MTData2Message *output, size_t output_size);

private:
  enum class State {
    WAIT_PREAMBLE1,
    WAIT_PREAMBLE2,
    WAIT_BID,
    WAIT_MID,
    WAIT_LEN,
    READ_DATA,
    WAIT_CHECKSUM,
    COMPLETE
  };

  static constexpr size_t MAX_MSG_LEN = 1024;
  static constexpr size_t MAX_MSG_DATA_LEN = 1024;
  struct RawMessageBuffer {
    uint8_t buffer[MAX_MSG_LEN];
    size_t len;
  };

  // Receive buffer for interrupt-driven reception
  static constexpr size_t RX_BUFFER_SIZE = 1024;

  UART_HandleTypeDef *uart_ = nullptr;

  static constexpr uint8_t PREAMBLE1 = 0xFA;
  static constexpr uint8_t PREAMBLE2 = 0xFF;
  static constexpr uint8_t MID_MTDATA2 = 0x36;

  RawMessageBuffer rxBuffer_{};

  static constexpr size_t MESSAGE_QUEUE_SIZE = 3;
  StaticQueue<RawMessageBuffer, MESSAGE_QUEUE_SIZE> rxMsgQueue_{};

  MTData2Message latestMessage_;
  
  // Static pointer to the active instance (for interrupt callback)
  static MTI300 *activeInstance_;

  MTI300::MTData2Message parseMTData2(const uint8_t *data, size_t len);

  void registerInstance();

};

} // TAUV
