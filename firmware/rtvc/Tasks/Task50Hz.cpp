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

bool Task50Hz::init(std::unique_ptr<Resources> resources) {
  resources_ = std::move(resources);
  eth_50hz_module.init(); // todo check retval
  esc_module.init(resources_->uarts);
  return true;
}

void Task50Hz::run() {
  eth_50hz_module.run();  // todo check retval
  esc_module.run();
}
