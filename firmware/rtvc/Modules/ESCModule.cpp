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
  return ModuleRunResult::OK;
}
