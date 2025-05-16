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
 #include "Logging.hpp"
 
 using namespace TAUV;
 
 ModuleInitResult ESCModule::init(const std::array<UART_HandleTypeDef *, Config::Thrusters::num_groups> &uarts) {
  LOG_INFO("ESCModule: Initializing with %zu UART groups", Config::Thrusters::num_groups);
  this->uarts = uarts;
 
  for (size_t group_idx = 0; group_idx < Config::Thrusters::num_groups; ++group_idx) {
    if (uarts[group_idx] == nullptr) {
      LOG_ERROR("ESCModule: UART handle for group %zu is null", group_idx);
      return ModuleInitResult::FATAL;
    }
    bool result = drivers[group_idx].setUART(uarts[group_idx]);
    if (!result) {
      LOG_ERROR("ESCModule: Failed to set UART for driver group %zu", group_idx);
      return ModuleInitResult::FATAL;
    }
    LOG_DEBUG("ESCModule: UART set for driver group %zu", group_idx);
  }
 
  // Verify ESCs are accessible
  LOG_INFO("ESCModule: Verifying ESC connectivity for %zu ESCs", Config::Thrusters::number_escs);
  size_t accessible_count = 0;
  
  for (size_t esc_idx = 0; esc_idx < Config::Thrusters::number_escs; ++esc_idx) {
    size_t group_idx = Config::Thrusters::esc_group_idx_map[esc_idx];
    size_t group_elem_idx = Config::Thrusters::esc_group_elem_idx_map[esc_idx];
    const auto& grp = Config::Thrusters::esc_groups[group_idx];
 
    bool result = drivers[group_idx].getFWversion(grp.vesc_ids[group_elem_idx]);
    if (!result) {
      LOG_ERROR("ESCModule: Failed to get firmware version for ESC %zu (group %zu, ID %d)", 
                esc_idx, group_idx, grp.vesc_ids[group_elem_idx]);
      // For now, we continue without returning FATAL, but this should be addressed
      // return ModuleInitResult::FATAL;
    } else {
      accessible_count++;
      // todo: verify fw compat and log version
    }
  }
  
  if (accessible_count < Config::Thrusters::number_escs) {
    LOG_WARNING("ESCModule: Only %zu of %zu ESCs are accessible", 
                accessible_count, Config::Thrusters::number_escs);
  } else {
    LOG_INFO("ESCModule: All %zu ESCs successfully initialized", Config::Thrusters::number_escs);
  }

  return ModuleInitResult::OK;
}

ModuleRunResult ESCModule::run() {
  // Get RPM values and enable flags from the input interface
  const auto& rpm_values = input_interface_.get_rpm();
  const auto& enable_flags = input_interface_.get_enable();

  bool all_commands_successful = true;

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

    // Check for abnormal RPM values that might indicate a problem
    if (std::abs(rpm_values[esc_idx]) > 10000) {  // Assuming 10000 RPM is an unreasonable value
      LOG_WARNING("ESCModule: Unusually high RPM requested for ESC %zu: %d RPM", 
                  esc_idx, rpm_values[esc_idx]);
    }

    // Set the RPM for the current ESC
    bool result = drivers[group_idx].setRPM(rpm_values[esc_idx], grp.vesc_ids[group_elem_idx]);
    if (!result) {
      LOG_ERROR("ESCModule: Failed to set RPM %d for ESC %zu (group %zu, ID %d)", 
                rpm_values[esc_idx], esc_idx, group_idx, grp.vesc_ids[group_elem_idx]);
      all_commands_successful = false;
    } else {
      LOG_DEBUG("ESCModule: Successfully set RPM %d for ESC %zu (group %zu, ID %d)",
                rpm_values[esc_idx], esc_idx, group_idx, grp.vesc_ids[group_elem_idx]);
    }
  }

  if (!all_commands_successful) {
    return ModuleRunResult::OUTPUT_INVALID;
  }

  return ModuleRunResult::OK;
}
