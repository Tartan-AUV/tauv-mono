#include <Arduino.h>
#include <TeensyTimerTool.h>
#include <ADC.h>
#include <chrono>

#include "config.h"
#include "frame.h"

#pragma once

namespace tx
{

    void setup(Config *config);
    void transmit(Frame *frame);
}