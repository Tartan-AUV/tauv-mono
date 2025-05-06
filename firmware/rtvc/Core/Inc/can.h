//
// Created by gleb on 4/11/25.
//

#ifndef CAN_H
#define CAN_H

#include "FreeRTOS.h"
#include "queue.h"
#include "stm32f7xx_hal.h"

/* Configuration */
#define CAN_100HZ_QUEUE_LENGTH 64

/* CAN Messsage IDs */

// XSens
// 400 Hz
#define CAN_MSG_ID_XSENS_SAMPLE_TIME 0x1
#define CAN_MSG_ID_XSENS_ORIENTATION_QUATERNION 0x2
#define CAN_MSG_ID_XSENS_RATE_OF_TURN 0x3
#define CAN_MSG_ID_XSENS_FREE_ACCELERATION 0x4
// 10 Hz
#define CAN_MSG_ID_XSENS_TEMPERATURE 0x5
#define CAN_MSG_ID_XSENS_PRESSURE 0x6

// VESC Base ID
#define VESC_BASE_CAN_ID 127

extern QueueHandle_t can100HzRxQueue;

/* CAN Parse */
typedef struct {
  CAN_RxHeaderTypeDef header;
  uint8_t data[8];
} CANRxMessage_t;

// Initialize the message queue(s)
void CAN_RxQueueInit();

// CAN ISR Override
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan);

#endif // CAN_H
