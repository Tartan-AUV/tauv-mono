/* TAUV RTVC */
/* 100Hz Ethernet Task */
/* Author: Gleb Ryabtsev */

/* INCLUDES */
#include "tasks.h"
#include "vehicle_config.h"

/* MESSAGE DEFINITIONS */

typedef struct {

} Eth100HzInputMessage;

typedef struct {
    // ESCs
    float esc_rpm[CONF_N_ESCS];
    bool esc_enable[CONF_N_ESCS];
    
    // ...
} Eth100HzMessage;


/* TASK DECLARATION */

RegularTaskStatus_t eth_100hz_task(const Eth100HzInputMessage *input_message, Eth100HzMessage *output_message);
