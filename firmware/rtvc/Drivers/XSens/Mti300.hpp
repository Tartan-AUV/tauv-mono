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

#include "Config.hpp"
#include "RingBuffer.hpp"
#include "StaticQueue.hpp"
#include "stm32f7xx_hal.h"

namespace TAUV {

class MTI300 {
 public:
  struct MTData2Message {
    std::optional<uint32_t> packetCounter;
    // Sample timestamp in nanoseconds
    std::optional<uint32_t> sampleTimeFine;
    // Quaternion
    std::optional<std::array<float, 4>> quaternion;
    // Free acceleration [ax, ay, az] in m/s²
    std::optional<std::array<float, 3>> freeAcceleration;
    // Angular velocity [gx, gy, gz] in rad/s
    std::optional<std::array<float, 3>> angularVelocity;
    // Temperature in C
    std::optional<float> temperature;
    // Pressure in Pa
    std::optional<float> pressure;
  };

  struct RawMessageBuffer {
    uint8_t data[Config::IMU::maxMessageDataSize];
    size_t len;
    uint8_t mid;
  };

  MTI300();

  void init(UART_HandleTypeDef *uart);

  void uartRxCallback();  // to be called from ISR

  uint8_t rxByte_ = 0;

  // Static pointer to the active instance (for interrupt callback)
  static MTI300 *activeInstance_;

  size_t processQueuedRawMessages(MTData2Message *output, size_t outputSize);

 private:
  enum class State {
    WAIT_PREAMBLE,
    WAIT_BID,
    WAIT_MID,
    WAIT_LEN,
    READ_DATA,
    WAIT_CHECKSUM,
  };

  UART_HandleTypeDef *uart_ = nullptr;

  static constexpr uint8_t PREAMBLE1 = 0xFA;
  static constexpr uint8_t BID = 0xFF;
  static constexpr uint8_t MID_MTDATA2 = 0x36;

  // Raw message parsing fields, updated in processCharacterFromISR()
  RawMessageBuffer rxBuffer_{};
  uint8_t dataIdx_ = 0;
  uint8_t checksum_ = 0;
  uint8_t dataLen_ = 0;
  State state_ = State::WAIT_PREAMBLE;

  StaticQueue<RawMessageBuffer, Config::IMU::queueLength> rxMsgQueue_{};

  static MTI300::MTData2Message parseMTData2(const RawMessageBuffer &buffer);
  void registerInstance();

  static inline float parseFloat(const uint8_t bytes[4]);
};

}  // namespace TAUV
