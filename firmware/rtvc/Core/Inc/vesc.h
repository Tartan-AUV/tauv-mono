//
// Created by Gleb Ryabtsev on 3/6/2025.
//

#ifndef VESC_H
#define VESC_H

#include <stddef.h>
#include <stdint.h>

#define VESC_SET_RPM_ID_MASK ((uint32_t) 0x00000300)

typedef enum
{
    VESC_OK,
    VESC_ERROR,
} VESCStatus_t;

typedef enum
{
    VESC_SET_RPM
} VESCCommand_t;

uint32_t vesc_get_can_msg_id(VESCCommand_t command, uint8_t esc_id);

VESCStatus_t vesc_get_duty_payload(float duty, uint8_t *buf, size_t size);

VESCStatus_t vesc_get_rpm_payload(uint32_t rpm, uint8_t *buf, size_t size);

VESCStatus_t vesc_get_current_payload(float current, uint8_t *buf, size_t size);

#endif //VESC_H
