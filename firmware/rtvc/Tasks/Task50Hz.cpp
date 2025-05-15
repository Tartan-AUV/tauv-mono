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

#include "Task50Hz.hpp"

#include "Eth50HzModule.hpp"

using namespace TAUV;

bool Task50Hz::init() {
  eth_50hz_module.init(); // todo check retval
  return true;
}

void TAUV::Task50Hz::run() {
  eth_50hz_module.run();  // todo check retval
}
