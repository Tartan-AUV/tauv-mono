#include <Arduino.h>
#include <TeensyTimerTool.h>

#include "config.h"
#include "tx.h"
#include "rx.h"

static Config config;
static Frame frame;

void setup()
{
  Serial.begin(115200);

  config.freq_lo = 50000;
  config.freq_hi = 51000;
  config.freq_sample = 200000;
  config.freq_bit = 10;
  config.freq_sync_bit = 20;
  config.sdft_N = 500;
  config.sdft_r = 0.99;

  tx::setup(&config);
  rx::setup(&config);

  Serial.println("initialized");

  frame = {
      1,
      {0},
      1,
      8,
  };
}

void loop()
{
  Serial.println("transmitting...");

  tx::transmit(&frame);

  ++frame.payload[0];

  Serial.println("done!");

  Serial.println("receiving...");

  rx::receive(&frame, 10 * ONE_MS_IN_NS);

  Serial.println("done!");
}
