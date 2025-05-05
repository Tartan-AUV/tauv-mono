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

/* Configuration */
#define ETH_50HZ_QUEUE_LENGTH 32

/* Message Definitions */

//typedef struct {
//    float EscRpm[CONF_N_ESCS];
//    bool EscEnable[CONF_N_ESCS];
//} Eth50HzInputMessage;

/* TASK DECLARATION */

// Initialize the message queue(s)
void Eth50Hz_RxQueueInit();

void Task_Eth50Hz_Init();

#endif // ETH50HZ_H
