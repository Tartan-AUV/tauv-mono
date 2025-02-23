//
// Created by Gleb Ryabtsev on 4/11/2025.
//

#include "can.h"
#include "queue.h"
#include "portmacro.h"

QueueHandle_t can100HzRxQueue;

static StaticQueue_t can100HzRxQueueStruct;
static uint8_t can100HzRxQueueBuffer[CAN_100HZ_QUEUE_LENGTH * sizeof(CANRxMessage_t)];

void CAN_RxQueueInit()
{

    can100HzRxQueue = xQueueCreateStatic(CAN_100HZ_QUEUE_LENGTH,
    				   	   	   	         sizeof(CANRxMessage_t),
										 can100HzRxQueueBuffer,
										 &can100HzRxQueueStruct);
}


void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan) {
	CANRxMessage_t msg;

    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &msg.header, msg.data) == HAL_OK)
    {
    	uint8_t devType = CAN_DEVICE_TYPE_FROM_ID(msg.header.ExtId);

    	BaseType_t xHigherPriorityTaskWoken;

    	switch (devType) {
    	case CAN_DEVICE_TYPE_XSENS:
    		xQueueSendToBackFromISR(can100HzRxQueue, &msg, &xHigherPriorityTaskWoken);
    		break;
    	case CAN_DEVICE_TYPE_VESC:
    		break;
    	default:
    		break;
    	}
    }
}
