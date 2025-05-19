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
#include "timers.h"

namespace TAUV {

// Forward declaration of TIM1 interrupt handler function
void MS5837_TIM1_OC_Callback();

class MS5837 {
 public:
  static const float Pa;
  static const float bar;
  static const float mbar;

  static const uint8_t MS5837_30BA;
  static const uint8_t MS5837_02BA;
  static const uint8_t MS5837_UNRECOGNISED;

  // Conversion time in timer ticks for maximum precision (8192 samples)
  static constexpr uint32_t CONVERSION_TIME_TICKS = 3000; // TIM1 ticks

  enum class Oversampling : uint8_t {
    MS5837_OS_256 = 0,
    MS5837_OS_512,
    MS5837_OS_1024,
    MS5837_OS_2048,
    MS5837_OS_4096,
    MS5837_OS_8192,
    Count
  };

  enum class ConversionState : uint8_t {
    IDLE,
    REQUESTING_TEMP,
    AWAITING_TEMP,
    READING_TEMP,
    REQUESTING_PRESSURE,
    AWAITING_PRESSURE,
    READING_PRESSURE,
  };

  static constexpr std::array<uint8_t, static_cast<size_t>(Oversampling::Count)>
      oversampling_command_map_pressure = {0x40, 0x42, 0x44, 0x46, 0x48, 0x4A};

  static constexpr std::array<uint8_t, static_cast<size_t>(Oversampling::Count)>
      oversampling_command_map_temperature = {0x50, 0x52, 0x54,
                                              0x56, 0x58, 0x5A};

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

  /** Start the conversion cycle by requesting temperature first
   * This function initiates I2C temperature conversion and starts TIM1
   */
  bool requestConversion(
      MS5837::Oversampling osr = Oversampling::MS5837_OS_8192);

  /** Returns true if both temperature and pressure conversions have completed
   * successfully
   */
  bool isDataReady() const { return data_ready; }

  bool isI2CError() {
    if (isr_error_flag_) {
      isr_error_flag_ = false;
      return true;
    }
    return false;
  }

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

  /** Timer interrupt callback handler for the conversion sequence
   */
  void timerCallback();
  void i2cMasterRxCpltCallback();
  void i2cMasterTxCpltCallback();
  void i2cMasterErrorCallback();

 private:
  enum class ConversionType : uint8_t {
    MS5837_PRESSURE,
    MS5837_TEMPERATURE,
    Count
  };

  // I2C handler
  I2C_HandleTypeDef *_hi2c;
  TIM_HandleTypeDef *_htim;

  ConversionState conversion_state = ConversionState::IDLE;
  Oversampling current_oversampling = Oversampling::MS5837_OS_8192;
  bool data_ready = false;  // Set when both temperature and pressure are valid
  bool isr_error_flag_ = false;  // Error flag
  uint8_t rx_data_[3]; // rx buffer
  uint8_t tx_data_;    // tx buffer

  uint16_t C[8];
  uint32_t D1_pres, D2_temp;
  int32_t TEMP;
  int32_t P;
  uint8_t _model;

  float fluidDensity;

  /** Performs calculations per the sensor data sheet for conversion and
   *  second order compensation.
   */
  void calculate();

  uint8_t crc4(uint16_t n_prom[]);

  /** Request temperature conversion
   */
  bool requestTemperature();

  /** Request pressure conversion
   */
  bool requestPressure();

  /** Read the result of a conversion
   */
  bool readConversion(MS5837::ConversionType type);

  friend void MS5837_TIM1_OC_Callback();
};

}  // namespace TAUV

#endif
