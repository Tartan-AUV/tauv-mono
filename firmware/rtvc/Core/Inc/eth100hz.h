/* TAUV RTVC */
/* 100Hz Ethernet Task */
/* Author: Gleb Ryabtsev */

/* INCLUDES */
#ifndef ETH100HZ_H
#define ETH100HZ_H

#include "messages.h"

/* MESSAGE DEFINITIONS */

typedef struct {

} Eth100HzInputMessage;


/* TASK DECLARATION */
void eth_100hz_task(const Eth100HzInputMessage* input_message, Eth100HzMessage* output_message);

#endif // ETH100HZ_H
