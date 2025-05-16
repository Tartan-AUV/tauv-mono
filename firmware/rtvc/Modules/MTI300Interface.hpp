/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Interface and message definitions for the MTI300 IMU module
 *
 *****************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include "Mti300.hpp"

namespace TAUV {

// Forward declaration of MTI300's MTData2Message struct
class MTI300;

// Message class to hold multiple MTData2 messages
struct MTI300Message {
  static constexpr size_t MAX_MESSAGES = 10;
  
  // Array of MTData2Message instances
  std::array<MTI300::MTData2Message, MAX_MESSAGES> messages;
  
  // Number of valid messages in the array
  size_t count = 0;
  
  // Timestamp when this message was updated
  uint32_t timestamp_ms = 0;
  
  // Clear all messages
  void clear() {
    count = 0;
  }
  
  // Add a new message if there's room
  bool addMessage(const MTI300::MTData2Message& msg) {
    if (count < MAX_MESSAGES) {
      messages[count++] = msg;
      return true;
    }
    return false;
  }
};

// Input interface for MTI300Module
class MTI300InputInterface {
public:
  MTI300InputInterface() = default;
  
  // Currently we don't need any configuration input for the MTI300 IMU
  // This interface can be extended in the future if needed
};

} // namespace TAUV
