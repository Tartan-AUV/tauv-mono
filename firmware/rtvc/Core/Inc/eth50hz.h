/* TAUV RTVC */
/* 50Hz Ethernet (Receive) Task */
/* Author: Victor Zayakov */

/* INCLUDES */
#ifndef ETH50HZ_H
#define ETH50HZ_H

#include "messages.h"
#include "FreeRTOS.h"
#include "queue.h"
#include "stm32f7xx_hal.h"

extern QueueHandle_t eth50HzRxQueue;

/* Message Definitions */

typedef struct {
	// Placeholder, change this code if you want to send messages to the Jetson
	// over Ethernet at 50Hz
	char placeholder;
} Eth50HzInputMessage;

/* TASK DECLARATION */

void Eth50Hz_RxQueueInit(); // Initialize the message queue(s)
void Task_Eth50Hz_Init();
void Task_Eth50Hz(const Eth50HzInputMessage *inputMessage, Eth50HzOutputMessage *outputMessage);

#endif // ETH50HZ_H
