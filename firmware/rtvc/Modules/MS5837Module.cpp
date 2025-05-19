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
  if (!sensor_driver_.requestConversion()) {
    LOG_ERROR("MS5837Module: Failed to request initial conversion");
    return ModuleInitResult::FATAL;
  }

  LOG_INFO("MS5837Module: Successfully initialized");
  return ModuleInitResult::OK;
}

ModuleRunResult MS5837Module::run() {
  // Clear previous message
  output_msg_.clear();

  // First read the result of the previous conversion
  bool read_success = sensor_driver_.read();
  
  // Then request a new conversion (it will alternate between pressure and temperature)
  bool request_success = sensor_driver_.requestConversion();
  
  if (!read_success) {
    // If this is the first run or if there was an error reading,
    // just schedule the next conversion and return
    static uint32_t last_warning_time = 0;
    uint32_t current_time = HAL_GetTick();
    if (current_time - last_warning_time > 5000) {  // Once every 5 seconds
      LOG_DEBUG("MS5837Module: No valid reading available");
      last_warning_time = current_time;
    }
    return ModuleRunResult::OUTPUT_INVALID;
  }
  
  if (!request_success) {
    LOG_ERROR("MS5837Module: Failed to request conversion");
    return ModuleRunResult::FATAL;
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
