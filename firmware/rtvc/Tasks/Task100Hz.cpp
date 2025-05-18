/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      Implementation of Task100Hz which includes MTI300 IMU processing
 *      and 100Hz ethernet communication
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
  auto mti_result = mti300_module_.init(resources_->imu_uart);
  if (mti_result != ModuleInitResult::OK) {
    LOG_ERROR("Task100Hz: Failed to initialize MTI300 module");
    return false;
  }
  
  // Initialize the Eth100Hz module
  auto eth_result = eth_100hz_module_.init();
  if (eth_result != ModuleInitResult::OK) {
    LOG_ERROR("Task100Hz: Failed to initialize Eth100Hz module");
    return false;
  }
  
  LOG_DEBUG("Task100Hz: Successfully initialized");
  return true;
}

void Task100Hz::run() {
  // Run the MTI300 module to collect IMU data
  auto mti_result = mti300_module_.run();
  
  if (mti_result == ModuleRunResult::FATAL) {
    LOG_ERROR("Task100Hz: MTI300 module encountered a fatal error");
  } else if (mti_result == ModuleRunResult::OK) {
    // Run the Eth100Hz module to send IMU data to Jetson
    auto eth_result = eth_100hz_module_.run();
    
    if (eth_result == ModuleRunResult::FATAL) {
      LOG_ERROR("Task100Hz: Eth100Hz module encountered a fatal error");
    }
  }
}
