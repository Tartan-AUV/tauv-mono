#include <Arduino.h>

#include "frame.h"

void Frame::update_checksum()
{
    this->checksum = 0;
}

bool Frame::check_checksum()
{
    return true;
}