/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Message definitions for ESC module with telemetry support
 *
 *****************************************************************************/
 
#pragma once

#include <array>
#include <cstdint>

#include "Config.hpp"

namespace TAUV {

struct ESCTelemetry {
  int32_t rpm;               // Motor RPM
  float voltage;             // Input voltage
  float current;             // Motor current
  float temperature_mosfet;  // MOSFET temperature
  float temperature_motor;   // Motor temperature
  uint8_t fault_code;        // ESC fault code
  bool data_valid;           // Indicates if telemetry data is valid
};

struct ESCMessage {
  // Array of telemetry data for each ESC
  std::array<ESCTelemetry, Config::Thrusters::number_escs> telemetry;
};

}  // namespace TAUV
