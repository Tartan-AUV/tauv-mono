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

extern UART_HandleTypeDef huart3;

namespace TAUV {

// Initialize static member
MTI300 *MTI300::activeInstance_ = nullptr;

MTI300::MTI300() {}

void MTI300::init(UART_HandleTypeDef *uart) {
  uart_ = uart;

  // Register this instance to receive callbacks
  registerInstance();

  while (HAL_UART_Receive_IT(uart_, &rxByte_, 1) != HAL_OK) {
    __HAL_UART_CLEAR_PEFLAG(
        uart_);  // Clears PE flag and also reads USART_SR & USART_DR
    __HAL_UART_CLEAR_FEFLAG(uart_);   // Clears FE
    __HAL_UART_CLEAR_NEFLAG(uart_);   // Clears NE
    __HAL_UART_CLEAR_OREFLAG(uart_);  // Clears ORE
    LOG_WARNING("MTi UART init error, retrying...");
    HAL_Delay(100);
  }
  LOG_DEBUG("MTi UART init success.");
}

void MTI300::registerInstance() { activeInstance_ = this; }

float MTI300::parseFloat(const uint8_t bytes[4]) {
  uint32_t as_int = (static_cast<uint32_t>(bytes[0]) << 24) |
                    (static_cast<uint32_t>(bytes[1]) << 16) |
                    (static_cast<uint32_t>(bytes[2]) << 8) |
                    (static_cast<uint32_t>(bytes[3]) << 0);

  float result;
  std::memcpy(&result, &as_int, sizeof(result));  // Avoids type punning
  return result;
}

void MTI300::uartRxCallback() {
  if (__HAL_UART_GET_FLAG(uart_, UART_FLAG_ORE) != RESET) {
#ifdef MTI_UART_ENABLE_DEBUG_PRINTS
    uint8_t dbg[] = "ISR ORE\n\r";
    HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
#endif

    __HAL_UART_CLEAR_OREFLAG(uart_);

  } else if (__HAL_UART_GET_FLAG(uart_, UART_FLAG_FE) != RESET) {
#ifdef MTI_UART_ENABLE_DEBUG_PRINTS
    uint8_t dbg[] = "ISR FE\n\r";
    HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
#endif

    __HAL_UART_CLEAR_FEFLAG(uart_);

  } else {
    switch (state_) {
      case State::WAIT_PREAMBLE:
        if (rxByte_ == PREAMBLE1) {
          checksum_ = 0;
          state_ = State::WAIT_BID;
        }
        break;
      case State::WAIT_BID:
        if (rxByte_ == BID) {
          checksum_ += rxByte_;
          state_ = State::WAIT_MID;
        } else {
          // reset
          state_ = State::WAIT_PREAMBLE;
          rxBuffer_.len = 0;
          checksum_ = 0;
        }
        break;
      case State::WAIT_MID:
        rxBuffer_.mid = rxByte_;
        checksum_ += rxByte_;
        state_ = State::WAIT_LEN;
        break;
      case State::WAIT_LEN:
        dataLen_ = rxByte_;
        checksum_ += rxByte_;
        rxBuffer_.len = 0;
        state_ = (dataLen_ > 0 && dataLen_ <= Config::IMU::maxMessageDataSize)
                     ? State::READ_DATA
                     : State::WAIT_CHECKSUM;
        break;
      case State::READ_DATA:
        rxBuffer_.data[rxBuffer_.len++] = rxByte_;
        checksum_ += rxByte_;
        if (rxBuffer_.len == dataLen_) {
          state_ = State::WAIT_CHECKSUM;
        }
        break;
      case State::WAIT_CHECKSUM:
#ifdef MTI_UART_ENABLE_DEBUG_PRINTS
      {
        uint8_t dbg[] = "ISR WC\n\r";
        HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
      }
#endif
        checksum_ += rxByte_;
        if (checksum_ == 0) {
#ifdef MTI_UART_ENABLE_DEBUG_PRINTS
          {
            uint8_t dbg[] = "ISR SUCC\n\r";
            HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
          }
#endif
          // Frame complete. Queue buffer for processing in the task
          rxMsgQueue_.sendFromISR(rxBuffer_, pdFALSE);
        }
        state_ = State::WAIT_PREAMBLE;  // expect the next message
        break;
    }
  }

  HAL_UART_Receive_IT(uart_, &rxByte_, 1);
}

MTI300::MTData2Message MTI300::parseMTData2(const RawMessageBuffer &buffer) {
  size_t i = 0;

  // Reset the message data for the new parsing
  MTData2Message msg;

  auto len = buffer.len;
  auto data = buffer.data;

  while (i + 3 <= len) {
    uint16_t data_id = (data[i] << 8) | data[i + 1];
    uint8_t data_size = data[i + 2];
    if (i + 3 + data_size > len) break;

    const uint8_t *field = &data[i + 3];

    switch (data_id) {
      case 0x1020: {  // Counter
        if (data_size == 4) {
          uint32_t ts =
              (field[0] << 24) | (field[1] << 16) | (field[2] << 8) | field[3];
          msg.packetCounter = ts;
        }
        break;
      }
      case 0x2010: {  // Quaternion [w, x, y, z]
        if (data_size == 16) {
          std::array<float, 4> q;
          for (int j = 0; j < 4; ++j) {
            q[j] = parseFloat(field + j * 4);
          }
          msg.quaternion = q;
        }
        break;
      }
      case 0x4030: {  // Free acceleration [ax, ay, az]
        if (data_size == 12) {
          std::array<float, 3> acc;
          for (int j = 0; j < 3; ++j) {
            acc[j] = parseFloat(field + j * 4);
          }
          msg.freeAcceleration = acc;
        }
        break;
      }
      case 0x8020: {  // Angular velocity [gx, gy, gz]
        if (data_size == 12) {
          std::array<float, 3> ang_vel;
          for (int j = 0; j < 3; ++j) {
            ang_vel[j] = parseFloat(field + j * 4);
          }
          msg.angularVelocity = ang_vel;
        }
        break;
      }
      case 0x1060: {  // Sample Time Fine
        if (data_size == 4) {
          uint32_t ts =
              (field[0] << 24) | (field[1] << 16) | (field[2] << 8) | field[3];
          msg.sampleTimeFine = ts;
        }
        break;
      }
      case 0x0810: {  // Temp C
        if (data_size == 4) {
          msg.temperature = parseFloat(field);
        }
        break;
      }
      case 0x3010: {  // Pressure Pa
        if (data_size == 4) {
          msg.pressure = parseFloat(field);
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

size_t MTI300::processQueuedRawMessages(MTData2Message *output,
                                        size_t outputSize) {
  RawMessageBuffer buffer{};
  size_t counter = 0;
  while (rxMsgQueue_.receive(buffer, 0)) {
    if (buffer.mid != MID_MTDATA2) {
      continue;
    }

    if (counter >= outputSize) {
      LOG_ERROR("IMU message overflow.");
      break;
    }

    if (buffer.len > 0) {
      auto msg = parseMTData2(buffer);
      if (outputSize > 0) {
        *output++ = msg;
        ++counter;
      }
    } else {
      LOG_ERROR("IMU message with zero length.");
    }
  }
  return counter;
}

}  // namespace TAUV
