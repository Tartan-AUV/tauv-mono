/* TAUV RTVC */
/* 100Hz Ethernet Task */
/* Author: Gleb Ryabtsev */

/* INCLUDES */
#ifndef ETH100HZ_H
#define ETH100HZ_H

#include "messages.h"

/* MESSAGE DEFINITIONS */

typedef struct {
    const XsensIMUFrame *imuFrames;
    const size_t nXsensImuFrames;
} Eth100HzInputMessage;


/* TASK DECLARATION */
void Task_Eth100Hz_Init();
void Task_Eth100Hz(const Eth100HzInputMessage* inputMessage, Eth100HzMessage* outputMessage);

#endif // ETH100HZ_H
