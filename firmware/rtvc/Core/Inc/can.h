//
// Created by gleb on 4/11/25.
//

#ifndef CAN_H
#define CAN_H

#include "stm32f7xx_hal.h"
#include "FreeRTOS.h"
#include "queue.h"

/* Configuration */
#define CAN_100HZ_QUEUE_LENGTH 64

// CAN Message ID structure
#define CAN_ID_DEVICE_TYPE_MSK 	0x000000F0
#define CAN_ID_DEVICE_ID_MSK 	0x0000000F
#define CAN_ID_MSG_TYPE_MSK		0x0000FF00

#define CAN_DEVICE_TYPE_FROM_ID(ext_id)	(uint8_t) ((0x000000F0 & (ext_id)) >> 4)
#define CAN_DEVICE_ID_FROM_ID(ext_id) 	(uint8_t)  (0x0000000F & (ext_id))
#define CAN_MSG_TYPE_FROM_ID(ext_id)	(uint8_t) ((0x0000FF00 & (ext_id)) >> 8)

// CAN Device types
#define CAN_DEVICE_TYPE_XSENS	0x1
#define CAN_DEVICE_TYPE_VESC    0x2

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

#endif //CAN_H
