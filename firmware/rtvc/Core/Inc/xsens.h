//
// Created by gleb on 4/11/25.
//

#ifndef XSENS_H
#define XSENS_H

#include "util.h"
#include "can.h"

// MT CAN message dictionary
#define CAN_MSG_TYPE_XSENS_SAMPLE_TIME 				0x81
#define CAN_MSG_TYPE_XSENS_ORIENTATION_QUATERNION 	0x82
#define CAN_MSG_TYPE_XSENS_RATE_OF_TURN 			0x83
#define CAN_MSG_TYPE_XSENS_FREE_ACCELERATION 		0x84

#define XSENS_IS_IMU_MSG(extId) (uint8_t)((extId) & 0x80)

#define CAN_XSENS_FRAME_N_MSGS	4
#define CAN_XSENS_IMU_FRAMES_MAX (CAN_100HZ_QUEUE_LENGTH / CAN_XSENS_FRAME_N_MSGS + 1)

typedef struct {
    Vec3_t      FreeAcceleration;
    Quat_t      Orientation;
    Vec3_t      RateOfTurn;
    uint64_t    SampleTime;
} XsensIMUFrame;

uint32_t   CAN_MsgParse_Xsens_SampleTime	(const uint8_t *data);
Quat_t     CAN_MsgParse_Xsens_Orientation	(const uint8_t *data);
Vec3_t     CAN_MsgParse_Xsens_RateOfTurn	(const uint8_t *data);
Vec3_t     CAN_MsgParse_Xsens_FreeAcceleration	(const uint8_t *data);

#endif //XSENS_H
