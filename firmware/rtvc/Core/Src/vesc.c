//
// Created by Gleb Ryabtsev on 3/6/2025.
//

#include "vesc.h"

uint32_t vesc_get_can_msg_id(const VESCCommand_t command, const uint8_t esc_id)
{
    switch (command)
    {
    case VESC_SET_RPM:
        return VESC_SET_RPM_ID_MASK | ((uint32_t) esc_id);
    default:
        return 0;
    }
}

VESCStatus_t vesc_get_duty_payload(float duty, uint8_t* buf, size_t size)
{
    return VESC_ERROR;
}

VESCStatus_t vesc_get_rpm_payload(uint32_t rpm, uint8_t* buf, size_t size)
{
    if (size != 4) {
        return VESC_ERROR;
    }

    uint32_t set_value = rpm;

    buf[0] = (set_value >> 24) & 0xFF;
    buf[1] = (set_value >> 16) & 0xFF;
    buf[2] = (set_value  >> 8  )  & 0xFF;
    buf[3] = set_value & 0xFF;

    return VESC_OK;
}

VESCStatus_t vesc_get_current_payload(float current, uint8_t* buf, size_t size)
{
    return VESC_ERROR;
}
