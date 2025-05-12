//
// Created by gleb on 4/11/25.
//

#ifndef XSENS_H
#define XSENS_H

#include "can.h"
#include "util.h"

// MT CAN message dictionary
#define CAN_XSENS_FRAME_N_MSGS 4
#define CAN_XSENS_IMU_FRAMES_MAX                                               \
  (CAN_100HZ_QUEUE_LENGTH / CAN_XSENS_FRAME_N_MSGS + 1)

typedef struct {
  Vec3_t FreeAcceleration;
  Quat_t Orientation;
  Vec3_t RateOfTurn;
  uint64_t SampleTime;
} XsensIMUFrame;

uint32_t CAN_MsgParse_Xsens_SampleTime(const uint8_t *data);
Quat_t CAN_MsgParse_Xsens_Orientation(const uint8_t *data);
Vec3_t CAN_MsgParse_Xsens_RateOfTurn(const uint8_t *data);
Vec3_t CAN_MsgParse_Xsens_FreeAcceleration(const uint8_t *data);

#endif // XSENS_H
