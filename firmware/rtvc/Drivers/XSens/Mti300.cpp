/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Driver for MTI300 IMU with interrupt-driven UART communication
 *
 *****************************************************************************/

#include "Mti300.hpp"
#include <cstring>
#include <cstdint>
#include "Logging.hpp"

namespace TAUV {

// Initialize static member
MTI300* MTI300::activeInstance_ = nullptr;

MTI300::MTI300() {}

void MTI300::init(UART_HandleTypeDef *uart) {
  uart_ = uart;
  
  // Reset the receive buffer
  rxHead_ = 0;
  rxTail_ = 0;
  bufferOverflow_ = false;
  
  // Register this instance to receive callbacks
  registerInstance();
  
  // Enable UART receive interrupt
  __HAL_UART_ENABLE_IT(uart_, UART_IT_RXNE);
}

void MTI300::registerInstance() {
  activeInstance_ = this;
}

void MTI300::uartRxCallback(UART_HandleTypeDef *huart) {
  // Check if there's an active instance
  if (activeInstance_ && activeInstance_->uart_ == huart) {
    // Read received byte
    if (__HAL_UART_GET_FLAG(huart, UART_FLAG_RXNE) != RESET) {
      uint8_t rx_byte = static_cast<uint8_t>(huart->Instance->RDR & 0xFF);
      activeInstance_->addByteToBuffer(rx_byte);
    }
    
    // Check for errors
    if (__HAL_UART_GET_FLAG(huart, UART_FLAG_ORE) != RESET) {
      LOG_WARNING("MTI300: UART Overrun Error detected");
      __HAL_UART_CLEAR_OREFLAG(huart);
    }
    
    if (__HAL_UART_GET_FLAG(huart, UART_FLAG_FE) != RESET) {
      LOG_WARNING("MTI300: UART Framing Error detected");
      __HAL_UART_CLEAR_FEFLAG(huart);
    }
  }
}

void MTI300::addByteToBuffer(uint8_t byte) {
  // Calculate the next head position
  size_t nextHead = (rxHead_ + 1) % RX_BUFFER_SIZE;
  
  // Check if buffer is full
  if (nextHead == rxTail_) {
    bufferOverflow_ = true;
    return;
  }
  
  // Add byte to buffer
  rxBuffer_[rxHead_] = byte;
  rxHead_ = nextHead;
}

void MTI300::processBuffer() {
  // Check if buffer overflow occurred
  if (bufferOverflow_) {
    LOG_WARNING("MTI300: Receive buffer overflow detected");
    bufferOverflow_ = false;
  }
  
  // Process all bytes in the buffer
  while (rxTail_ != rxHead_) {
    uint8_t byte = rxBuffer_[rxTail_];
    rxTail_ = (rxTail_ + 1) % RX_BUFFER_SIZE;
    
    // Process this byte
    processByte(byte);
  }
}

void MTI300::resetParser() {
  state_ = State::WAIT_PREAMBLE1;
  dataIdx_ = 0;
  checksum_ = 0;
}

void MTI300::processByte(uint8_t byte) {
  switch (state_) {
    case State::WAIT_PREAMBLE1:
      if (byte == PREAMBLE1) {
        checksum_ = byte;
        state_ = State::WAIT_PREAMBLE2;
      }
      break;
    case State::WAIT_PREAMBLE2:
      if (byte == PREAMBLE2) {
        checksum_ ^= byte;
        state_ = State::WAIT_BID;
      } else {
        resetParser();
      }
      break;
    case State::WAIT_BID:
      checksum_ ^= byte;
      state_ = State::WAIT_MID;
      break;
    case State::WAIT_MID:
      checksum_ ^= byte;
      if (byte == MID_MTDATA2) {
        state_ = State::WAIT_LEN;
      } else {
        resetParser();
      }
      break;
    case State::WAIT_LEN:
      dataLen_ = byte;
      checksum_ ^= byte;
      dataIdx_ = 0;
      state_ = (dataLen_ > 0 && dataLen_ <= MAX_MSG_LEN) ? State::READ_DATA : State::WAIT_CHECKSUM;
      break;
    case State::READ_DATA:
      buffer_[dataIdx_++] = byte;
      checksum_ ^= byte;
      if (dataIdx_ == dataLen_) {
        state_ = State::WAIT_CHECKSUM;
      }
      break;
    case State::WAIT_CHECKSUM:
      if (checksum_ == byte) {
        parseMTData2(buffer_, dataLen_);
      }
      resetParser();
      break;
  }
}

void MTI300::parseMTData2(const uint8_t *data, size_t len) {
  size_t i = 0;

  // Reset the message data for the new parsing
  MTData2Message newMessage;

  while (i + 3 <= len) {
    uint16_t data_id = (data[i] << 8) | data[i + 1];
    uint8_t data_size = data[i + 2];
    if (i + 3 + data_size > len) break;

    const uint8_t *field = &data[i + 3];

    switch (data_id) {
      case 0x2010: {  // Quaternion [w, x, y, z]
        if (data_size == 16) {
          std::array<float, 4> q;
          std::memcpy(q.data(), field, sizeof(q));
          newMessage.quaternion = q;
        }
        break;
      }
      case 0x4020: {  // Free acceleration [ax, ay, az]
        if (data_size == 12) {
          std::array<float, 3> acc;
          std::memcpy(acc.data(), field, sizeof(acc));
          newMessage.freeAcceleration = acc;
        }
        break;
      }
      case 0x8020: {  // Angular velocity [gx, gy, gz]
        if (data_size == 12) {
          std::array<float, 3> gyro;
          std::memcpy(gyro.data(), field, sizeof(gyro));
          newMessage.angularVelocity = gyro;
        }
        break;
      }
      case 0x1060: {  // Sample Time Fine
        if (data_size == 4) {
          uint32_t ts = (field[0] << 24) | (field[1] << 16) |
                        (field[2] << 8) | field[3];
          newMessage.sampleTimeFine = ts;
        }
        break;
      }
      default:
        // Unknown or unhandled field
        break;
    }

    i += 3 + data_size;
  }
  
  // Only update the latest message if we successfully parsed some data
  if (newMessage.quaternion.has_value() || 
      newMessage.freeAcceleration.has_value() || 
      newMessage.angularVelocity.has_value() || 
      newMessage.sampleTimeFine.has_value()) {
    latestMessage_ = std::move(newMessage);
  }
}

} // TAUV