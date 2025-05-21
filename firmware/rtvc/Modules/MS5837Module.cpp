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
  sensor_driver_.requestConversion();
  // conversion starts in a separate thread, so we just pray

  LOG_INFO("MS5837Module: Successfully initialized");
  return ModuleInitResult::OK;
}

ModuleRunResult MS5837Module::run() {
  // Clear previous message
  output_msg_.clear();

  auto result = ModuleRunResult::OK;

  bool conversion_valid = sensor_driver_.calculate();
  sensor_driver_.requestConversion(); // don't fuck around and request next conversion asap

  if (conversion_valid) {
    uint32_t current_time = HAL_GetTick();
    // Update the message with the sensor data
    output_msg_.pressure = sensor_driver_.pressure();
    output_msg_.temperature = sensor_driver_.temperature();
    output_msg_.depth = sensor_driver_.depth();
    output_msg_.valid = true;
    output_msg_.timestamp_ms = HAL_GetTick();
    LOG_DEBUG("MS5837Module: Pressure: %.2f mbar, Temperature: %.2f C, Depth: %.2f m",
             output_msg_.pressure, output_msg_.temperature, output_msg_.depth);
  } else {
    result = ModuleRunResult::OUTPUT_INVALID;
    LOG_DEBUG("No depth reading...");
  }

  return result;
}

}  // namespace TAUV
