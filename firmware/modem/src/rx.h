#include <Arduino.h>
#include <TeensyTimerTool.h>
#include <ADC.h>

#include "config.h"
#include "frame.h"

#pragma once

namespace rx
{
    void setup(Config *config);
    void receive(Frame *frame, std::chrono::nanoseconds timeout);
}