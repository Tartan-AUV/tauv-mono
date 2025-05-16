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
  LOG_INFO("Task100Hz: Initializing");
  
  if (!resources || !resources->imu_uart) {
    LOG_ERROR("Task100Hz: Missing IMU UART handle in resources");
    return false;
  }
  
  resources_ = std::move(resources);
  
  // Initialize the MTI300 module
  auto result = mti300_module_.init(resources_->imu_uart);
  if (result != ModuleInitResult::OK) {
    LOG_ERROR("Task100Hz: Failed to initialize MTI300 module");
    return false;
  }
  
  LOG_INFO("Task100Hz: Successfully initialized");
  return true;
}

void Task100Hz::run() {
  // Run the MTI300 module
  auto result = mti300_module_.run();
  
  if (result == ModuleRunResult::OUTPUT_INVALID) {
    LOG_DEBUG("Task100Hz: MTI300 module produced no valid output");
  } else if (result == ModuleRunResult::FATAL) {
    LOG_ERROR("Task100Hz: MTI300 module encountered a fatal error");
  } else {
    // Process the latest IMU data if needed
    const auto& imu_msg = mti300_msg_;
    LOG_DEBUG("Task100Hz: Received %u IMU messages", imu_msg.count);
    
    // Example of accessing the data (can be removed or modified as needed)
    if (imu_msg.count > 0) {
      const auto& latest_imu_data = imu_msg.messages[0];
      if (latest_imu_data.quaternion.has_value()) {
        const auto& quat = latest_imu_data.quaternion.value();
        LOG_DEBUG("Task100Hz: Latest quaternion: [%f, %f, %f, %f]", 
                 quat[0], quat[1], quat[2], quat[3]);
      }
    }
  }
  
  // Add other 100Hz tasks here as needed
}
