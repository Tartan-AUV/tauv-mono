/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Implementation of MS5837 Pressure/Temperature Sensor Module with I2C
 *      Uses blocking delays instead of ISRs for conversion timing
 *
 *****************************************************************************/

#include "MS5837Module.hpp"
#include "Logging.hpp"
#include "Config.hpp"
#include "cmsis_os.h"

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

  // Start the first conversion cycle
  if (!sensor_driver_.requestTemperatureConversion()) {
    LOG_WARN("MS5837Module: Failed to start initial temperature conversion");
  }

  LOG_INFO("MS5837Module: Successfully initialized");
  return ModuleInitResult::OK;
}

ModuleRunResult MS5837Module::run() {
  // Clear previous message
  output_msg_.clear();

  auto result = ModuleRunResult::OK;
  
  // Get the conversion delay for the current OSR setting
  uint32_t conversion_delay_ms = sensor_driver_.getConversionDelayMs();
  
  // ==== Conversion Sequence ====
  // The sequence is:
  // 1. Request temperature conversion
  // 2. Wait for conversion
  // 3. Read temperature data
  // 4. Request pressure conversion
  // 5. Wait for conversion
  // 6. Read pressure data
  // 7. Calculate compensated values

  // Step 1: Request temperature conversion
  if (!sensor_driver_.requestTemperatureConversion()) {
    LOG_WARN("MS5837Module: Failed to start temperature conversion for next cycle");
  }

  // Step 2: Wait for temperature conversion to complete
  osDelay(conversion_delay_ms);
  
  // Step 3: Read temperature data
  if (!sensor_driver_.readTemperature()) {
    LOG_WARN("MS5837Module: Failed to read temperature");
    result = ModuleRunResult::OUTPUT_INVALID;
    // Try to restart the conversion cycle for next time
    sensor_driver_.requestTemperatureConversion();
    return result;
  }
  
  // Step 4: Request pressure conversion
  if (!sensor_driver_.requestPressureConversion()) {
    LOG_WARN("MS5837Module: Failed to request pressure conversion");
    result = ModuleRunResult::OUTPUT_INVALID;
    // Try to restart the conversion cycle for next time
    sensor_driver_.requestTemperatureConversion();
    return result;
  }
  
  // Step 5: Wait for pressure conversion to complete
  osDelay(conversion_delay_ms);
  
  // Step 6: Read pressure data
  if (!sensor_driver_.readPressure()) {
    LOG_WARN("MS5837Module: Failed to read pressure");
    result = ModuleRunResult::OUTPUT_INVALID;
    // Try to restart the conversion cycle for next time
    sensor_driver_.requestTemperatureConversion();
    return result;
  }
  
  // Step 7: Calculate compensated pressure and temperature
  if (sensor_driver_.calculate()) {
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
    LOG_DEBUG("MS5837Module: No valid data for calculation");
  }
  
  return result;
}

}  // namespace TAUV