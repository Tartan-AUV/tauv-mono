/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/17/25
 *
 *  Description:
 *      Contains user-defined callbacks for HAL ISRs.
 *
 *****************************************************************************/

#include "Config.hpp"
#include "Mti300.hpp"

using namespace TAUV;
/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      Custom User Interrupts
 *  Date:        Current Date
 *
 *  Description:
 *      Custom interrupt handlers for the project
 *
 *****************************************************************************/

#include "MS5837.h"
#include "stm32f7xx_hal.h"

// External TIM handle declaration
extern TIM_HandleTypeDef htim1;

extern "C" {

#include "main.h"
#include "stm32f7xx_hal.h"

void HAL_TIM_OC_DelayElapsedCallback(TIM_HandleTypeDef *htim) {
  // Time-keeping for time stamping
  if (htim->Instance == TIM5 && htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1) {
    // timestamp_secs += 1;

    uint32_t next = __HAL_TIM_GET_COUNTER(htim) - 1000000;
    __HAL_TIM_SET_COUNTER(htim, next);
  } else if (htim->Instance == TIM1 &&
             htim->Channel == HAL_TIM_ACTIVE_CHANNEL_1 &&
             MS5837::activeInstance_ != nullptr) {
    MS5837::activeInstance_->conversionTimerISRCallback();
  }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart->Instance == Config::IMU::uartInstance) {
    MTI300::activeInstance_->uartRxCallback();
  }
}

void HAL_I2C_MasterTxCpltCallback(I2C_HandleTypeDef *hi2c) {
  if (hi2c->Instance == I2C1 && MS5837::activeInstance_ != nullptr) {
    MS5837::activeInstance_->i2cTxCpltISRCallback();
  }
}

void HAL_I2C_MasterRxCpltCallback(I2C_HandleTypeDef *hi2c) {
  if (hi2c->Instance == I2C1 && MS5837::activeInstance_ != nullptr) {
    MS5837::activeInstance_->i2cRxCpltISRCallback();
  }
}

void HAL_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c) {
  if (hi2c->Instance == I2C1 && MS5837::activeInstance_ != nullptr) {
    MS5837::activeInstance_->i2cErrorISRCallback();
  }
}
}