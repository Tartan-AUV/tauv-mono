/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Interface definitions for the MTI300 IMU module
 *
 *****************************************************************************/

#pragma once

namespace TAUV {

// Input interface for MTI300Module
class MTI300InputInterface {
 public:
  MTI300InputInterface() = default;

  // Currently we don't need any configuration input for the MTI300 IMU
  // This interface can be extended in the future if needed
};

}  // namespace TAUV
