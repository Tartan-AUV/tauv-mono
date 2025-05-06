/* TAUV RTVC */
/* 50Hz CAN Task */
/* Author: Victor Zayakov */



/* INCLUDES */
#ifndef CAN50HZ_H
#define CAN50HZ_H

#include "messages.h"
#include "eth50hz.h"

/* MESSAGE DEFINITIONS */

typedef struct {
    Eth50HzOutputMessage outputMsg;
} CAN50HzInputMessage;

/* TASK DECLARATION */

void Task_CAN50Hz(const CAN50HzInputMessage* inputMsg, CAN50HzMessage* outputMsg);

#endif // CAN100HZ_H
