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
#include "ESCMessage.hpp"
#include "Logging.hpp"
#include "main.h"  // For GPIO pin definitions

using namespace TAUV;

 ModuleInitResult ESCModule::init(const std::array<UART_HandleTypeDef *, Config::Thrusters::num_groups> &uarts) {
  LOG_INFO("ESCModule: Initializing with %d UART groups", Config::Thrusters::num_groups);
  this->uarts = uarts;

  // Wait for ESCs to initialize
  if (Config::Thrusters::init_delay_ms > 0) {
    osDelay(Config::Thrusters::init_delay_ms);
  }

  for (size_t group_idx = 0; group_idx < Config::Thrusters::num_groups; ++group_idx) {
    if (uarts[group_idx] == nullptr) {
      LOG_ERROR("ESCModule: UART handle for group %d is null", group_idx);
      return ModuleInitResult::FATAL;
    }
    bool result = vesc_interfaces_[group_idx].setUART(uarts[group_idx]);
    if (!result) {
      LOG_ERROR("ESCModule: Failed to set UART for driver group %d", group_idx);
      return ModuleInitResult::FATAL;
    }
    LOG_DEBUG("ESCModule: UART set for driver group %d", group_idx);
  }

  // Verify ESCs are accessible
  LOG_INFO("ESCModule: Verifying ESC connectivity for %d ESCs", Config::Thrusters::number_escs);
  size_t accessible_count = 0;

  for (size_t esc_idx = 0; esc_idx < Config::Thrusters::number_escs; ++esc_idx) {
    size_t group_idx = Config::Thrusters::esc_group_idx_map[esc_idx];
    size_t group_elem_idx = Config::Thrusters::esc_group_elem_idx_map[esc_idx];
    const auto& grp = Config::Thrusters::esc_groups[group_idx];

    bool result = false;
    // Check if this ESC is directly connected to UART
    if (grp.vesc_ids[group_elem_idx] == grp.uart_connected_id) {
      // Use the overload without canId for directly connected ESC
      result = vesc_interfaces_[group_idx].getFWversion();
      LOG_DEBUG("ESCModule: Using direct UART communication for ESC %d (group %d, ID %d)",
               esc_idx, group_idx, grp.vesc_ids[group_elem_idx]);
    } else {
      // Use the overload with canId for ESCs accessed via CAN bus
      result = vesc_interfaces_[group_idx].getFWversion(grp.vesc_ids[group_elem_idx]);
      LOG_DEBUG("ESCModule: Using CAN communication for ESC %d (group %d, ID %d)",
               esc_idx, group_idx, grp.vesc_ids[group_elem_idx]);
    }

    if (!result) {
      LOG_ERROR("ESCModule: Failed to get firmware version for ESC %d (group %d, ID %d)",
                esc_idx, group_idx, grp.vesc_ids[group_elem_idx]);
      // For now, we continue without returning FATAL, but this should be addressed
      // return ModuleInitResult::FATAL;
    } else {
      accessible_count++;

      // Log and check firmware version
      auto& fw_version = vesc_interfaces_[group_idx].fw_version;
      LOG_INFO("ESCModule: ESC %d (group %d, ID %d) firmware version: %d.%d",
              esc_idx, group_idx, grp.vesc_ids[group_elem_idx],
              fw_version.major, fw_version.minor);

      // Check if firmware version matches expected version
      if (fw_version.major != Config::Thrusters::expected_fw_major ||
          fw_version.minor != Config::Thrusters::expected_fw_minor) {
        LOG_WARN("ESCModule: ESC %d (group %d, ID %d) has unexpected firmware version %d.%d (expected %d.%d)",
                    esc_idx, group_idx, grp.vesc_ids[group_elem_idx],
                    fw_version.major, fw_version.minor,
                    Config::Thrusters::expected_fw_major, Config::Thrusters::expected_fw_minor);
      }
    }
  }

  if (accessible_count < Config::Thrusters::number_escs) {
    LOG_WARN("ESCModule: Only %d of %d ESCs are accessible",
                accessible_count, Config::Thrusters::number_escs);
  } else {
    LOG_INFO("ESCModule: All %d ESCs successfully initialized", Config::Thrusters::number_escs);
  }

  return ModuleInitResult::OK;
}

ModuleRunResult ESCModule::run() {
  // Check killswitch state - if low (GPIO_PIN_RESET), latch the killswitch
  GPIO_PinState killswitch_state = HAL_GPIO_ReadPin(GPIO_KILLSWITCH_IN_GPIO_Port, GPIO_KILLSWITCH_IN_Pin);
  
  // Once the killswitch goes low, latch it permanently (until system reset)
  if (killswitch_state == GPIO_PIN_RESET && !killswitch_latched_) {
    killswitch_latched_ = true;
    LOG_WARN("ESCModule: Killswitch activated and LATCHED - all ESCs will remain disabled until system reset");
  }
  
  // Log current state periodically for debugging
  static uint32_t log_counter = 0;
  if (killswitch_latched_ && (++log_counter % 50) == 0) {  // Log every 50 cycles (1 second at 50Hz)
    LOG_DEBUG("ESCModule: Killswitch remains latched - sending zero RPM to all ESCs");
  }

  // Process commands for each ESC
  for (size_t i = 0; i < Config::Thrusters::number_escs; i++) {
    // Get group index and element index for this ESC
    size_t group_idx = Config::Thrusters::esc_group_idx_map[i];
    size_t elem_idx = Config::Thrusters::esc_group_elem_idx_map[i];

    // Get VESC ID for this ESC
    uint8_t vesc_id = Config::Thrusters::esc_groups[group_idx].vesc_ids[elem_idx];

    // Check if this ESC requires direct UART communication
    bool is_uart_connected = (vesc_id == Config::Thrusters::esc_groups[group_idx].uart_connected_id);

    // Determine the RPM to send based on killswitch latch and enable state
    float rpm_command = 0;
    
    // Only send non-zero RPM if killswitch is NOT latched
    if (!killswitch_latched_ && input_interface_.is_valid() && input_interface_.get_enable()[i]) {
      rpm_command = input_interface_.get_rpm()[i];
    }
    
    // Send the RPM command to the ESC
    if (is_uart_connected) {
      vesc_interfaces_[group_idx].setRPM(rpm_command);
    } else {
      // For CAN-connected ESCs, we need to specify the CAN ID
      vesc_interfaces_[group_idx].setRPM(rpm_command, vesc_id);
    }

    // Collect telemetry data from ESCs
    if (Config::Thrusters::collect_telemetry) {
      if (is_uart_connected) {
        // Only request telemetry from directly connected ESCs
        bool got_values = vesc_interfaces_[group_idx].getVescValues();

        if (got_values) {
          // Store telemetry data for this ESC
          const auto& data = vesc_interfaces_[group_idx].data;
          output_msg_.telemetry[i].rpm = data.rpm;
          output_msg_.telemetry[i].voltage = data.inpVoltage;
          output_msg_.telemetry[i].current = data.avgMotorCurrent;
          output_msg_.telemetry[i].temperature_mosfet = data.tempMosfet;
          output_msg_.telemetry[i].fault_code = static_cast<uint8_t>(data.error);
          output_msg_.telemetry[i].data_valid = true;
        } else {
          // Mark this ESC's telemetry as invalid
          output_msg_.telemetry[i].data_valid = false;
        }
      } else {
        // For CAN-connected ESCs, request telemetry with their CAN ID
        bool got_values = vesc_interfaces_[group_idx].getVescValues(vesc_id);

        if (got_values) {
          // Store telemetry data for this ESC
          const auto& data = vesc_interfaces_[group_idx].data;
          output_msg_.telemetry[i].rpm = data.rpm;
          output_msg_.telemetry[i].voltage = data.inpVoltage;
          output_msg_.telemetry[i].current = data.avgMotorCurrent;
          output_msg_.telemetry[i].temperature_mosfet = data.tempMosfet;
          output_msg_.telemetry[i].fault_code = static_cast<uint8_t>(data.error);
          output_msg_.telemetry[i].data_valid = true;
        } else {
          // Mark this ESC's telemetry as invalid
          output_msg_.telemetry[i].data_valid = false;
        }
      }
    }
  }


  return ModuleRunResult::OK;
}
