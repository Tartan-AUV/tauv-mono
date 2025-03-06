/* TAUV RTVC */
/* 100Hz CAN Task */
/* Author: Gleb Ryabtsev */

/* INCLUDES */
#include "tasks.h"
#include "eth100hz.h"

/* MESSAGE DEFINITIONS */

typedef struct {
    const Eth100HzMessage *eth_100hz_msg;
} CAN100HzInputMessage;

typedef struct {

} CAN100HzMessage;


/* TASK DECLARATION */

RegularTaskStatus_t can_100hz_task(const CAN100HzInputMessage *input_msg, CAN100HzMessage *output_msg);

