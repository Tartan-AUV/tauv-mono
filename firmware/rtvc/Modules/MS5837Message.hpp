/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Message definitions for the MS5837 pressure/temperature sensor module
 *
 *****************************************************************************/

#pragma once

#include <cstdint>

namespace TAUV {

// Message class to hold MS5837 data
struct MS5837Message {
  // Pressure in mbar (default scale)
  float pressure = 0.0f;
  
  // Temperature in degrees Celsius
  float temperature = 0.0f;
  
  // Depth in meters (calculated from pressure)
  float depth = 0.0f;
  
  // Whether the data is valid
  bool valid = false;
  
  // Timestamp when this message was updated
  uint32_t timestamp_ms = 0;
  
  // Clear the message
  void clear() { 
    pressure = 0.0f;
    temperature = 0.0f;
    depth = 0.0f;
    valid = false;
  }
};

}  // namespace TAUV
