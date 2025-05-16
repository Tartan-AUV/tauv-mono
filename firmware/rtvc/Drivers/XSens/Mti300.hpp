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

#include "stm32f7xx_hal.h"
#include <optional>
#include <array>
#include <cstddef>

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

  void processByte(uint8_t byte);  // Call from UART RX handler
  
  // Access the latest parsed MTData2 message
  const MTData2Message& getLatestMessage() const { return latestMessage_; }
  
  // Process any pending bytes in the receive buffer
  void processBuffer();
  
  // Static handler for UART RX interrupt
  static void uartRxCallback(UART_HandleTypeDef *huart);
  
  // Register this instance to receive UART interrupt callbacks
  void registerInstance();

private:
  enum class State {
    WAIT_PREAMBLE1,
    WAIT_PREAMBLE2,
    WAIT_BID,
    WAIT_MID,
    WAIT_LEN,
    READ_DATA,
    WAIT_CHECKSUM
  };

  // Receive buffer for interrupt-driven reception
  static constexpr size_t RX_BUFFER_SIZE = 256;
  uint8_t rxBuffer_[RX_BUFFER_SIZE];
  volatile size_t rxHead_ = 0;
  volatile size_t rxTail_ = 0;
  bool bufferOverflow_ = false;

  UART_HandleTypeDef *uart_ = nullptr;
  State state_ = State::WAIT_PREAMBLE1;

  static constexpr uint8_t PREAMBLE1 = 0xFA;
  static constexpr uint8_t PREAMBLE2 = 0xFF;
  static constexpr uint8_t MID_MTDATA2 = 0x36;

  static constexpr size_t MAX_MSG_LEN = 255;
  uint8_t buffer_[MAX_MSG_LEN];
  uint8_t dataLen_ = 0;
  uint8_t dataIdx_ = 0;
  uint8_t checksum_ = 0;
  
  MTData2Message latestMessage_;
  
  // Static pointer to the active instance (for interrupt callback)
  static MTI300* activeInstance_;

  void resetParser();
  void parseMTData2(const uint8_t *data, size_t len);
  
  // Add a byte to the receive buffer (called from interrupt)
  void addByteToBuffer(uint8_t byte);
};

} // TAUV
