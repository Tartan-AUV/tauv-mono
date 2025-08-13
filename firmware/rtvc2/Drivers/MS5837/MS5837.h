/* Blue Robotics Arduino MS5837-30BA Pressure/Temperature Sensor Library
------------------------------------------------------------

Title: Blue Robotics Arduino MS5837-30BA Pressure/Temperature Sensor Library

Description: This library provides utilities to communicate with and to
read data from the Measurement Specialties MS5837-30BA pressure/temperature
sensor.

Authors: Rustom Jehangir, Blue Robotics Inc.
         Adam Šimko, Blue Robotics Inc.
         Modified for STM32 by TartanAUV team

-------------------------------
The MIT License (MIT)

Copyright (c) 2015 Blue Robotics Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-------------------------------*/

#ifndef MS5837_H_BLUEROBOTICS
#define MS5837_H_BLUEROBOTICS

#include <array>

#include "FreeRTOS.h"
#include "stm32f7xx_hal.h"

namespace TAUV {

class MS5837 {
 public:
  static constexpr float Pa = 100.0f;
  static constexpr float bar = 0.001f;
  static constexpr float mbar = 1.0f;
  static constexpr uint8_t MS5837_30BA = 0;
  static constexpr uint8_t MS5837_02BA = 1;
  static constexpr uint8_t MS5837_UNRECOGNISED = 255;

  MS5837();
  ~MS5837();

  bool init(I2C_HandleTypeDef *hi2c);
  bool begin(I2C_HandleTypeDef *hi2c);  // Calls init()

  /** Set model of MS5837 sensor. Valid options are MS5837::MS5837_30BA
   * (default) and MS5837::MS5837_02BA.
   */
  void setModel(uint8_t model);
  uint8_t getModel();

  /** Provide the density of the working fluid in kg/m^3. Default is for
   * seawater. Should be 997 for freshwater.
   */
  void setFluidDensity(float density);

  /** Request temperature conversion (D2)
   * Sends the command to start temperature conversion
   */
  bool requestTemperatureConversion();

  /** Read temperature data after conversion
   * Must be called after appropriate delay from requestTemperatureConversion()
   */
  bool readTemperature();

  /** Request pressure conversion (D1)
   * Sends the command to start pressure conversion
   */
  bool requestPressureConversion();

  /** Read pressure data after conversion
   * Must be called after appropriate delay from requestPressureConversion()
   */
  bool readPressure();

  /** Calculate compensated pressure and temperature from raw values
   * Call this after reading both temperature and pressure
   * Returns true if calculation was successful
   */
  bool calculate();

  /** Check if both temperature and pressure data are ready for calculation
   */
  bool isDataReady() const { return data_ready_; }

  /** Get the conversion delay needed for the current OSR setting
   * Returns delay in milliseconds
   */
  uint32_t getConversionDelayMs() const;

  /** Pressure returned in mbar or mbar*conversion rate.
   */
  float pressure(float conversion = 1.0f);

  /** Temperature returned in deg C.
   */
  float temperature();

  /** Depth returned in meters (valid for operation in incompressible
   *  liquids only. Uses density that is set for fresh or seawater.
   */
  float depth();

  /** Altitude returned in meters (valid for operation in air only).
   */
  float altitude();



 private:
  enum class OverSamplingRatio : uint8_t {
    MS5837_OS_256 = 0,
    MS5837_OS_512,
    MS5837_OS_1024,
    MS5837_OS_2048,
    MS5837_OS_4096,
    MS5837_OS_8192,
    Count
  };


  static constexpr uint8_t MS5837_ADDR = 0x76;
  static constexpr uint8_t MS5837_RESET = 0x1E;
  static constexpr uint8_t MS5837_ADC_READ = 0x00;
  static constexpr uint8_t MS5837_PROM_READ = 0xA0;
  static constexpr uint8_t MS5837_CONVERT_D1_8192 = 0x4A;
  static constexpr uint8_t MS5837_CONVERT_D2_8192 = 0x5A;
  static constexpr uint8_t MS5837_02BA01 =
      0x00;  // Sensor version: From MS5837_02BA datasheet Version PROM Word 0
  static constexpr uint8_t MS5837_02BA21 =
      0x15;  // Sensor version: From MS5837_02BA datasheet Version PROM Word 0
  static constexpr uint8_t MS5837_30BA26 =
      0x1A;  // Sensor version: From MS5837_30BA datasheet Version PROM Word 0
  static constexpr OverSamplingRatio OSR = OverSamplingRatio::MS5837_OS_512;
  
  // Conversion delays in milliseconds for each OSR setting
  static constexpr std::array<uint32_t, static_cast<size_t>(OverSamplingRatio::Count)>
      conversion_delays_ms = {1, 2, 3, 5, 9, 18};

  static constexpr std::array<uint8_t, static_cast<size_t>(OverSamplingRatio::Count)>
      oversampling_command_map_pressure = {0x40, 0x42, 0x44, 0x46, 0x48, 0x4A};

  static constexpr std::array<uint8_t, static_cast<size_t>(OverSamplingRatio::Count)>
      oversampling_command_map_temperature = {0x50, 0x52, 0x54,
                                              0x56, 0x58, 0x5A};

  // State variables
  bool data_ready_ = false;  // Set when both temperature and pressure are valid, reset on calculate
  bool temp_ready_ = false;  // Temperature data has been read
  bool pressure_ready_ = false;  // Pressure data has been read

  uint16_t C[8];
  uint32_t D1_pres, D2_temp;
  int32_t TEMP;
  int32_t P;
  uint8_t _model;

  float fluidDensity;

  // Handles
  I2C_HandleTypeDef *hi2c_;

  uint8_t crc4(uint16_t n_prom[]);
};

}  // namespace TAUV

#endif
