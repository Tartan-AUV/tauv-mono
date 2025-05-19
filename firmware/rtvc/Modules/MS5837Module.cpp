/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Implementation of MS5837 Pressure/Temperature Sensor Module with I2C
 *
 *****************************************************************************/

#include "MS5837Module.hpp"
#include "Logging.hpp"
#include "Config.hpp"

namespace TAUV {

ModuleInitResult MS5837Module::init(I2C_HandleTypeDef *hi2c) {
  LOG_INFO("MS5837Module: Initializing...");

  if (hi2c == nullptr) {
    LOG_ERROR("MS5837Module: I2C handle is null");
    return ModuleInitResult::FATAL;
  }

  hi2c_ = hi2c;

  // Initialize the sensor
  if (!sensor_driver_.init(hi2c_)) {
    LOG_ERROR("MS5837Module: Failed to initialize the sensor");
    return ModuleInitResult::FATAL;
  }

  // Set the fluid density to freshwater (default is seawater)
  sensor_driver_.setFluidDensity(997.0f); // 997 kg/m^3 for freshwater

  // Request initial conversion to start the measurement cycle
  if (!sensor_driver_.requestConversion(Config::Depth::osr)) {
    LOG_ERROR("MS5837Module: Failed to request initial conversion");
    return ModuleInitResult::FATAL;
  }

  LOG_INFO("MS5837Module: Successfully initialized");
  return ModuleInitResult::OK;
}

ModuleRunResult MS5837Module::run() {
  // Clear previous message
  output_msg_.clear();

  // Check if sensor data is ready and read it
  bool has_valid_data = sensor_driver_.read();
  
  // If we don't have valid data or if we need to start a new conversion cycle
  if (!sensor_driver_.isDataReady()) {
    // Request a new conversion if we're not already in the middle of one
    if (sensor_driver_.requestConversion(Config::Depth::osr)) {
      LOG_DEBUG("MS5837Module: Started new conversion cycle");
    } else {
      static uint32_t last_error_time = 0;
      uint32_t current_time = HAL_GetTick();
      if (current_time - last_error_time > 5000) {  // Limit error logging to once every 5 seconds
        LOG_ERROR("MS5837Module: Failed to request conversion");
        last_error_time = current_time;
      }
    }
  }
  
  if (!has_valid_data) {
    static uint32_t last_warning_time = 0;
    uint32_t current_time = HAL_GetTick();
    if (current_time - last_warning_time > 5000) {  // Once every 5 seconds
      LOG_DEBUG("MS5837Module: No valid reading available");
      last_warning_time = current_time;
    }
    return ModuleRunResult::OUTPUT_INVALID;
  }

  // Update the message with the sensor data
  output_msg_.pressure = sensor_driver_.pressure();
  output_msg_.temperature = sensor_driver_.temperature();
  output_msg_.depth = sensor_driver_.depth();
  output_msg_.valid = true;
  output_msg_.timestamp_ms = HAL_GetTick();

  LOG_DEBUG("MS5837Module: Pressure: %.2f mbar, Temperature: %.2f C, Depth: %.2f m",
           output_msg_.pressure, output_msg_.temperature, output_msg_.depth);
  
  return ModuleRunResult::OK;
}

}  // namespace TAUV
