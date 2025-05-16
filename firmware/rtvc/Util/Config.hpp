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

extern "C" {
#include "stm32f7xx_hal.h"
}

#include <array>

namespace TAUV::Config {

namespace Thrusters {

static constexpr size_t number_escs = 4;
static constexpr size_t escs_per_group_max = 4;

struct ESC_Group {
  std::array<uint8_t, escs_per_group_max> vesc_ids;
};

static constexpr std::array<ESC_Group, 2> esc_groups{
  // ESC_Group{
  //   .vesc_ids = {128, 129, 130, 131}
  // },
  ESC_Group{
    .vesc_ids = {132, 133, 134, 135}
  },
};

static constexpr size_t num_groups = esc_groups.size();

static constexpr std::array<size_t, number_escs> esc_group_idx_map{
  0, 0, 0, 0 // , 1, 1, 1, 1
};

static constexpr std::array<size_t, number_escs> esc_group_elem_idx_map{
  0, 1, 2, 3 // , 0, 1, 2, 3
};

}  // namespace Thrusters

namespace Network {

static constexpr uint32_t jetson_1hz_port = 11001;
static constexpr uint32_t jetson_10hz_port = 11002;
static constexpr uint32_t jetson_50hz_port = 11004;
static constexpr uint32_t jetson_100hz_port = 11003;
static constexpr uint32_t jetson_log_port = 11010;

}  // namespace Network

};  // namespace TAUV::Config
