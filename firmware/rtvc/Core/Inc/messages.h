//
// Created by Gleb Ryabtsev on 3/6/2025.
//

#ifndef MESSAGES_H
#define MESSAGES_H

#include <stdbool.h>
#include <stdint.h>

#include "vehicle_config.h"
#include "util.h"
#include "xsens.h"

typedef struct {
    XsensIMUFrame   ImuFrames[CAN_XSENS_IMU_FRAMES_MAX];
    size_t          NImuFrames;
} CAN100HzMessage;

void CAN100HzMessage_Init(CAN100HzMessage *m);

typedef struct {
    // ESCs
    float EscRpm[CONF_N_ESCS];
    bool EscEnable[CONF_N_ESCS];
    // And more...

} Eth100HzMessage;

typedef struct {
    // ESCs
    float EscRpm[CONF_N_ESCS];
    bool EscEnable[CONF_N_ESCS];

} Eth50HzMessage;

typedef struct {
	float EscRpm;
	bool EscEnable;
} CAN50HzMessage;

typedef struct {
    float Pressure;
    float Temperature;
} XsensEnvironmentalMessage;


#endif //MESSAGES_H
