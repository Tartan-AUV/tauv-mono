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

typedef struct {
    // ESCs
    int32_t EscRpm[CONF_N_ESCS];
    bool EscEnable[CONF_N_ESCS];
    // And more...

} Eth100HzMessage;

typedef struct {
    // ESCs
    int32_t EscRpm[CONF_N_ESCS];
    bool EscEnable[CONF_N_ESCS];

} Eth50HzMessage;

typedef struct {
	// Placeholder code
	// Fill this struct if you want to receive data on RTVC
	// over CAN at 50HZ
	char placeholder;
} CAN50HzMessage;

typedef struct {
    float Pressure;
    float Temperature;
} XsensEnvironmentalMessage;

/* Configuration */
#define ETH_50HZ_QUEUE_LENGTH 32

typedef struct {
	Eth50HzMessage OutMsgs[ETH_50HZ_QUEUE_LENGTH];
	size_t NumEscMsgs;
} Eth50HzOutputMessage;


void CAN100HzMessage_Init(CAN100HzMessage *m);
void Eth50HzMessage_Init(Eth50HzOutputMessage *m);

#endif //MESSAGES_H
