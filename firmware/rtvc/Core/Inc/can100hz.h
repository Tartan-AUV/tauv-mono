/* TAUV RTVC */
/* 100Hz CAN Task */
/* Author: Gleb Ryabtsev */



/* INCLUDES */
#ifndef CAN100HZ_H
#define CAN100HZ_H

#include "messages.h"

/* MESSAGE DEFINITIONS */

typedef struct {
    const Eth100HzMessage *eth_100hz_msg;
} CAN100HzInputMessage;

/* TASK DECLARATION */

void Task_CAN100Hz_Init();
void Task_CAN100Hz(const CAN100HzInputMessage* inputMsg, CAN100HzMessage* outputMsg);

#endif // CAN100HZ_H
