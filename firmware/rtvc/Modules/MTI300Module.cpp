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
  
  // Process any bytes received via interrupt
  static constexpr size_t MAX_MSGS = 3;
  MTI300::MTData2Message msgs[MAX_MSGS]; // todo: make same as ISR queue len and MTIModuleMessage arr size
  size_t n_msgs = imu_driver_.processQueuedMessages(msgs, MAX_MSGS);

  for (size_t i = 0; i < n_msgs; i++) {
    auto &msg = msgs[i];
    // Only add the message if it contains at least one valid field
    if (msg.quaternion.has_value() ||
        msg.freeAcceleration.has_value() ||
        msg.angularVelocity.has_value() ||
        msg.sampleTimeFine.has_value()) {
      if (!output_msg_.addMessage(msg)) {
        LOG_WARNING("MTI300Module: Message buffer full, discarding new message");
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

} // namespace TAUV
