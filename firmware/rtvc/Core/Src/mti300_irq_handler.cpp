/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      Handler to bridge between C IRQ handlers and C++ MTI300 driver
 *
 *****************************************************************************/

#include "stm32f7xx_hal.h"
#include "Mti300.hpp"

// This function is called from the UART IRQ handler in stm32f7xx_it.c
extern "C" void MTI300_UART_IRQHandler(UART_HandleTypeDef *huart) {
  // Call the static UART RX callback method in MTI300
  TAUV::MTI300::uartRxCallback(huart);
}
