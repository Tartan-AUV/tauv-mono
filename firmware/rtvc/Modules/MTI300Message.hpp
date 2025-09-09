/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Message definitions for the MTI300 IMU module
 *
 *****************************************************************************/

#pragma once

#include <array>
#include <cstddef>

#include "Mti300.hpp"

namespace TAUV {

// Message class to hold multiple MTData2 messages
struct MTI300Message {
  static constexpr size_t MAX_MESSAGES = 10;

  // Array of MTData2Message instances
  std::array<MTI300::MTData2Message, MAX_MESSAGES> frames;

  // Number of valid messages in the array
  size_t count = 0;

  // Timestamp when this message was updated
  uint32_t timestamp_ms = 0;

  // Clear all messages
  void clear() { count = 0; }

  // Add a new message if there's room
  bool addMessage(const MTI300::MTData2Message& msg) {
    if (count < MAX_MESSAGES) {
      frames[count++] = msg;
      return true;
    }
    return false;
  }
};

}  // namespace TAUV
