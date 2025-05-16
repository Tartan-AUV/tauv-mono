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

namespace TAUV {

ModuleInitResult MTI300Module::init(UART_HandleTypeDef *uart) {
  LOG_INFO("MTI300Module: Initializing");
  
  if (uart == nullptr) {
    LOG_ERROR("MTI300Module: UART handle is null");
    return ModuleInitResult::FATAL;
  }
  
  uart_ = uart;
  
  // Initialize the IMU driver with UART
  imu_driver_.init(uart_);
  
  // Configure UART for interrupt-driven reception
  // Note: We need to register the UART RX callback in the UART HAL IRQ handler
  // This is typically done in stm32f7xx_it.c in the UART IRQ handler
  
  // Enable UART receive interrupt
  __HAL_UART_ENABLE_IT(uart_, UART_IT_RXNE);
  
  LOG_INFO("MTI300Module: Successfully initialized with interrupt-driven UART");
  return ModuleInitResult::OK;
}

ModuleRunResult MTI300Module::run() {
  // Record current time for this message
  output_msg_.timestamp_ms = HAL_GetTick();
  
  // Clear previous messages
  output_msg_.clear();
  
  // Process any bytes received via interrupt
  imu_driver_.processBuffer();
  
  // Always add the latest message from the driver
  const auto& latest_msg = imu_driver_.getLatestMessage();
  
  // Only add the message if it contains at least one valid field
  if (latest_msg.quaternion.has_value() || 
      latest_msg.freeAcceleration.has_value() || 
      latest_msg.angularVelocity.has_value() || 
      latest_msg.sampleTimeFine.has_value()) {
    if (!output_msg_.addMessage(latest_msg)) {
      LOG_WARNING("MTI300Module: Message buffer full, discarding new message");
    }
  }
  
  if (output_msg_.count == 0) {
    LOG_DEBUG("MTI300Module: No valid messages in buffer");
    return ModuleRunResult::OUTPUT_INVALID;
  }
  
  return ModuleRunResult::OK;
}

} // namespace TAUV
