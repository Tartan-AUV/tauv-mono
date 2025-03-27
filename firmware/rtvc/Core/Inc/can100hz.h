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

void can_100hz_task(const CAN100HzInputMessage* input_msg, CAN100HzMessage* output_msg);

#endif // CAN100HZ_H
