/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/
 #include "ESCModule.hpp"

using namespace TAUV;

ModuleInitResult ESCModule::init(const std::array<UART_HandleTypeDef *, Config::Thrusters::num_groups> &uarts) {
  this->uarts = uarts;

  for (size_t group_idx = 0; group_idx < Config::Thrusters::num_groups; ++group_idx) {
	  drivers[group_idx].setUART(uarts[group_idx]);
  }

  // Verify ESCs are accessible
  for (size_t esc_idx = 0; esc_idx < Config::Thrusters::number_escs; ++esc_idx) {
    size_t group_idx = Config::Thrusters::esc_group_idx_map[esc_idx];
    size_t group_elem_idx = Config::Thrusters::esc_group_elem_idx_map[esc_idx];
    const auto& grp = Config::Thrusters::esc_groups[group_idx];

    bool result = drivers[group_idx].getFWversion(grp.vesc_ids[group_elem_idx]);
    if (!result) {
      // todo error
//      return ModuleInitResult::FATAL;
    }
    // todo: verify fw compat and log version
  }

  return ModuleInitResult::OK;
}

ModuleRunResult ESCModule::run() {
  // Get RPM values and enable flags from the input interface
  const auto& rpm_values = input_interface_.get_rpm();
  const auto& enable_flags = input_interface_.get_enable();

  // Iterate through all ESCs and set their RPM if enabled
  for (size_t esc_idx = 0; esc_idx < Config::Thrusters::number_escs; ++esc_idx) {
    // Skip if ESC is not enabled
    if (!enable_flags[esc_idx]) {
      continue;
    }

    // Get the group index and element index for the current ESC
    size_t group_idx = Config::Thrusters::esc_group_idx_map[esc_idx];
    size_t group_elem_idx = Config::Thrusters::esc_group_elem_idx_map[esc_idx];
    const auto& grp = Config::Thrusters::esc_groups[group_idx];

    // Set the RPM for the current ESC
    drivers[group_idx].setRPM(rpm_values[esc_idx], grp.vesc_ids[group_elem_idx]);
  }


  return ModuleRunResult::OK;
}
