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

namespace TAUV::Config {

namespace Thrusters {

static constexpr size_t number_escs = 8;

}

namespace Network {

static constexpr uint32_t jetson_1hz_port = 11001;
static constexpr uint32_t jetson_10hz_port = 11002;
static constexpr uint32_t jetson_50hz_port = 11004;
static constexpr uint32_t jetson_100hz_port = 11003;
static constexpr uint32_t jetson_log_port = 11010;

}

};
