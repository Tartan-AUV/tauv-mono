/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Implementation of MTI300 IMU Module with interrupt-driven UART
 *
 *****************************************************************************/

#include "MTI300Module.hpp"

#include "Logging.hpp"

extern UART_HandleTypeDef huart3;

namespace TAUV {

ModuleInitResult MTI300Module::init(UART_HandleTypeDef *uart) {
  LOG_INFO("MTI300Module: Initializing...");

  if (uart == nullptr) {
    LOG_ERROR("MTI300Module: UART handle is null");
    return ModuleInitResult::FATAL;
  }

  uart_ = uart;

  imu_driver_.init(uart_);

  LOG_INFO("MTI300Module: Successfully initialized");
  return ModuleInitResult::OK;
}

ModuleRunResult MTI300Module::run() {
  // Clear previous messages
  output_msg_.clear();

  // Check if there's a UART error, clear flags, and restart RX transactions
  if (uart_->ErrorCode != HAL_UART_ERROR_NONE) {
    // UART is in error state
    if (uart_->ErrorCode & HAL_UART_ERROR_ORE) {
      uint8_t dbg[] = "ORE\n\r";
      HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
      // Overrun Error
    }
    if (uart_->ErrorCode & HAL_UART_ERROR_FE) {
      uint8_t dbg[] = "FE\n\r";
      HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
      // Framing Error
    }
    if (uart_->ErrorCode & HAL_UART_ERROR_PE) {
      uint8_t dbg[] = "PE\n\r";
      HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
      // Parity Error
    }
    if (uart_->ErrorCode & HAL_UART_ERROR_NE) {
      uint8_t dbg[] = "PE\n\r";
      HAL_UART_Transmit(&huart3, dbg, sizeof(dbg), HAL_MAX_DELAY);
      // Noise Error
    }
    __HAL_UART_CLEAR_PEFLAG(
        uart_);  // Clears PE flag and also reads USART_SR & USART_DR
    __HAL_UART_CLEAR_FEFLAG(uart_);   // Clears FE
    __HAL_UART_CLEAR_NEFLAG(uart_);   // Clears NE
    __HAL_UART_CLEAR_OREFLAG(uart_);  // Clears ORE
    imu_driver_.init(uart_);
  }

  // Process any bytes received via interrupt
  MTI300::MTData2Message
      mt_data_msgs[Config::IMU::queueLength];
  size_t n_msgs = imu_driver_.processQueuedRawMessages(mt_data_msgs, Config::IMU::queueLength);
  LOG_DEBUG("Received %d messages", n_msgs);
  for (size_t i = 0; i < n_msgs; i++) {
    auto &msg = mt_data_msgs[i];
    // Only add the message if it contains at least one valid field
    if (msg.quaternion.has_value() || msg.freeAcceleration.has_value() ||
        msg.angularVelocity.has_value() || msg.sampleTimeFine.has_value()) {
      if (!output_msg_.addMessage(msg)) {
        LOG_WARNING(
            "MTI300Module: Message buffer full, discarding new message");
      }
    }
  }

  if (n_msgs == 0) {
    // Only log this occasionally to avoid spamming
    static uint32_t last_warning_time = 0;
    uint32_t current_time = HAL_GetTick();
    if (current_time - last_warning_time > 5000) {  // Once every 5 seconds
      LOG_DEBUG("MTI300Module: No valid messages in buffer");
      // note this can rarely happen if there's timing misalignment
      last_warning_time = current_time;
    }
    return ModuleRunResult::OUTPUT_INVALID;
  }

  return ModuleRunResult::OK;
}

}  // namespace TAUV
