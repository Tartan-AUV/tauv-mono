//
// Created by gleb on 4/11/25.
//

#include "xsens.h"

#define ORIENTATION_SCALING (1.0f / 32767.0f)
#define RATE_OF_TURN_SCALING (1.0f / 512.0f)
#define ACCELERATION_SCALING (1.0f / 256.0f)

uint32_t CAN_MsgParse_Xsens_SampleTime(const uint8_t *data)
{
    return *(uint32_t*) data;
}

Quat_t CAN_MsgParse_Xsens_Orientation(const uint8_t *data)
{
    Quat_t q;
    q.w = ((float) *(int16_t*)&data[0]) * ORIENTATION_SCALING;
    q.x = ((float) *(int16_t*)&data[2]) * ORIENTATION_SCALING;
    q.y = ((float) *(int16_t*)&data[4]) * ORIENTATION_SCALING;
    q.z = ((float) *(int16_t*)&data[6]) * ORIENTATION_SCALING;
    return q;
}

Vec3_t CAN_MsgParse_Xsens_RateOfTurn(const uint8_t *data)
{
    Vec3_t v;
    v.x = ((float) *(int16_t*)&data[0]) * RATE_OF_TURN_SCALING;
    v.y = ((float) *(int16_t*)&data[2]) * RATE_OF_TURN_SCALING;
    v.z = ((float) *(int16_t*)&data[4]) * RATE_OF_TURN_SCALING;
    return v;
}
Vec3_t CAN_MsgParse_Xsens_FreeAcceleration(const uint8_t *data)
{
    Vec3_t v;
    v.x = ((float) *(int16_t*)&data[0]) * ACCELERATION_SCALING;
    v.y = ((float) *(int16_t*)&data[2]) * ACCELERATION_SCALING;
    v.z = ((float) *(int16_t*)&data[4]) * ACCELERATION_SCALING;
    return v;
}
