#include <Arduino.h>

#pragma once

#define FRAME_MAX_PAYLOAD_LENGTH 8
#define FRAME_META_LENGTH 2

struct Frame
{
    uint8_t payload_length;
    uint8_t payload[FRAME_MAX_PAYLOAD_LENGTH];
    uint8_t checksum;

    void update_checksum();
    bool check_checksum();
};
