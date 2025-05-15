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

ModuleInitResult ESCModule::init(UART_HandleTypeDef *left_uart, UART_HandleTypeDef *right_uart) {
  this->left_uart = left_uart;
  this->right_uart = right_uart;
}
ModuleRunResult ESCModule::run() {

}
