/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Interface definitions for the MS5837 pressure/temperature sensor module
 *
 *****************************************************************************/

#pragma once

namespace TAUV {

// Input interface for MS5837Module
class MS5837InputInterface {
 public:
  MS5837InputInterface() = default;

  // Currently we don't need any configuration input for the MS5837 sensor
  // This interface can be extended in the future if needed
};

}  // namespace TAUV
