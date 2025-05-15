/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/

#pragma once

#include <string>
#include <utility>

using std::size_t;

#include "Singleton.hpp"

namespace TAUV {

enum class ModuleInitResult {
  OK,
  FATAL
};

enum class ModuleRunResult {
  OK,
  OUTPUT_INVALID,
  FATAL
};

template <typename I, typename M>
class ModuleBase {
public:
  virtual const char* getName() const = 0;
  virtual float getFrequency() const = 0;

  ModuleBase(const I &input_interface, M &output_msg) :
    input_interface_(input_interface), output_msg_(output_msg) {};

  virtual ModuleRunResult run() = 0;

  M get_output_message() {
    return output_msg_;
  }

protected:
  const I &input_interface_ = nullptr;
  M &output_msg_ = nullptr;
};

}
