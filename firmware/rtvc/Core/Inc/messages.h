//
// Created by Gleb Ryabtsev on 3/6/2025.
//

#ifndef MESSAGES_H
#define MESSAGES_H

#include <stdbool.h>

#include "vehicle_config.h"

typedef struct {

} CAN100HzMessage;

typedef struct {
    // ESCs
    float esc_rpm[CONF_N_ESCS];
    bool esc_enable[CONF_N_ESCS];

    // ...
} Eth100HzMessage;


#endif //MESSAGES_H
