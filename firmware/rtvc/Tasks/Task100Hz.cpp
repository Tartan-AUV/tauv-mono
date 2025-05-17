/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      Implementation of Task100Hz which includes MTI300 IMU processing
 *
 *****************************************************************************/

#include "Task100Hz.hpp"
#include "Logging.hpp"

using namespace TAUV;

bool Task100Hz::init(std::unique_ptr<Resources> resources) {
  LOG_DEBUG("Task100Hz: Initializing");

  if (!resources || !resources->imu_uart) {
    return false;
  }
  
  resources_ = std::move(resources);
  
  // Initialize the MTI300 module
  auto result = mti300_module_.init(resources_->imu_uart);
  if (result != ModuleInitResult::OK) {
    return false;
  }
  
  LOG_DEBUG("Task100Hz: Successfully initialized");
  return true;
}

void Task100Hz::run() {
  // Run the MTI300 module
  auto result = mti300_module_.run();
  
  if (result == ModuleRunResult::FATAL) {
    LOG_ERROR("Task100Hz: MTI300 module encountered a fatal error");
  } else {
    // Process the latest IMU data
    const auto& imu_msg = mti300_msg_;
  }
  
  // Add other 100Hz tasks here as needed
}
