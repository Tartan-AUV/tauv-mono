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

#include <cstdint>
#include <cstring>

#include "Config.hpp"
#include "Logging.hpp"

namespace TAUV {

// Initialize static member
MTI300* MTI300::activeInstance_ = nullptr;

MTI300::MTI300() {}

void MTI300::init(UART_HandleTypeDef *uart) {
  uart_ = uart;

  // Register this instance to receive callbacks
  registerInstance();
  
  auto res = HAL_UARTEx_ReceiveToIdle_DMA(uart_, rxBuffer_.buffer, MAX_MSG_LEN);
  if (res != HAL_OK) {
    LOG_ERROR("MTI300: Failed to initialize UART");
  }
  // HAL_UARTEx_ReceiveToIdle_DMA(uart_, rxBuffer_.buffer, MAX_MSG_LEN);
}

void MTI300::registerInstance() {
  activeInstance_ = this;
}

void MTI300::uartRxCallback(size_t len) {
  if (activeInstance_ != nullptr) {
    // Check for errors
    if (__HAL_UART_GET_FLAG(activeInstance_->uart_, UART_FLAG_ORE) != RESET) {
      // LOG_WARNING("MTI300: UART Overrun Error detected");
      __HAL_UART_CLEAR_OREFLAG(activeInstance_->uart_);
    } else if (__HAL_UART_GET_FLAG(activeInstance_->uart_, UART_FLAG_FE) != RESET) {
      // LOG_WARNING("MTI300: UART Framing Error detected");
      __HAL_UART_CLEAR_FEFLAG(activeInstance_->uart_);
    } else {
      // TODO: push buffer to queue
      activeInstance_->rxBuffer_.len = len;
      activeInstance_->rxMsgQueue_.sendFromISR(activeInstance_->rxBuffer_, pdFALSE);
    }

    HAL_UART_Receive_IT(activeInstance_->uart_, activeInstance_->rxBuffer_.buffer, MAX_MSG_LEN);
  }
}

size_t MTI300::processQueuedMessages(MTData2Message *output,
                                     size_t output_size) {
  uint8_t byte;
  __HAL_UART_CLEAR_FLAG(uart_, UART_CLEAR_PEF | UART_CLEAR_FEF | UART_CLEAR_NEF | UART_CLEAR_OREF);
  auto res = HAL_UARTEx_ReceiveToIdle_DMA(uart_, rxBuffer_.buffer, MAX_MSG_LEN);

  State state = State::WAIT_PREAMBLE1;
  uint8_t dataLen = 0;
  uint8_t dataIdx = 0;
  uint8_t checksum = 0;

  uint8_t dataBuffer[MAX_MSG_DATA_LEN];

  RawMessageBuffer msg{};
  size_t counter = 0;
  while (rxMsgQueue_.receive(msg, 0)) {
    for (size_t i = 0; i < msg.len; ++i) {
      uint8_t byte = msg.buffer[i];
      switch (state) {
        case State::WAIT_PREAMBLE1:
          if (byte == PREAMBLE1) {
            checksum = byte;
            state = State::WAIT_PREAMBLE2;
          }
          break;
        case State::WAIT_PREAMBLE2:
          if (byte == PREAMBLE2) {
            checksum ^= byte;
            state = State::WAIT_BID;
          } else {
            // reset
            state = State::WAIT_PREAMBLE1;
            dataIdx = 0;
            checksum = 0;
          }
          break;
        case State::WAIT_BID:
          checksum ^= byte;
          state = State::WAIT_MID;
          break;
        case State::WAIT_MID:
          checksum ^= byte;
          if (byte == MID_MTDATA2) {
            state = State::WAIT_LEN;
          } else {
            // reset
            state = State::WAIT_PREAMBLE1;
            dataIdx = 0;
            checksum = 0;
          }
          break;
        case State::WAIT_LEN:
          dataLen = byte;
          checksum ^= byte;
          dataIdx = 0;
          state = (dataLen > 0 && dataLen <= MAX_MSG_LEN)
                      ? State::READ_DATA
                      : State::WAIT_CHECKSUM;
          break;
        case State::READ_DATA:
          dataBuffer[dataIdx++] = byte;
          checksum ^= byte;
          if (dataIdx == dataLen) {
            state = State::WAIT_CHECKSUM;
          }
          break;
        case State::WAIT_CHECKSUM:
          if (checksum == byte) {
            auto mtData2Msg = parseMTData2(dataBuffer, dataLen);
            output[counter++] = mtData2Msg;
          }
          state = State::COMPLETE;
          break;
        case State::COMPLETE:
          LOG_WARNING("MTI300: Message parsing error.");
      }
    }
  }

  return counter;
}

MTI300::MTData2Message MTI300::parseMTData2(const uint8_t *data, size_t len) {
  size_t i = 0;

  // Reset the message data for the new parsing
  MTData2Message msg;

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
          msg.quaternion = q;
        }
        break;
      }
      case 0x4020: {  // Free acceleration [ax, ay, az]
        if (data_size == 12) {
          std::array<float, 3> acc;
          std::memcpy(acc.data(), field, sizeof(acc));
          msg.freeAcceleration = acc;
        }
        break;
      }
      case 0x8020: {  // Angular velocity [gx, gy, gz]
        if (data_size == 12) {
          std::array<float, 3> gyro;
          std::memcpy(gyro.data(), field, sizeof(gyro));
          msg.angularVelocity = gyro;
        }
        break;
      }
      case 0x1060: {  // Sample Time Fine
        if (data_size == 4) {
          uint32_t ts = (field[0] << 24) | (field[1] << 16) |
                        (field[2] << 8) | field[3];
          msg.sampleTimeFine = ts;
        }
        break;
      }
      default:
        // Unknown or unhandled field
        break;
    }

    i += 3 + data_size;
  }

  return msg;
}

extern "C" void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t size)
{
  if (huart->Instance == Config::IMU::uart_instance) {
    // Forward the callback to the MTI300 handler
    MTI300::uartRxCallback(size);
  }
}

} // TAUV
