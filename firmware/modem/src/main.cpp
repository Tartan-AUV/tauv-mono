#include <Arduino.h>
#include <TeensyTimerTool.h>

#include "config.h"
#include "tx.h"
#include "rx.h"

static Config config;
static Frame tx_frame;
static Frame rx_frame;

void setup()
{
  Serial.begin(115200);
  Serial.setTimeout(100);

  config.freq_lo = 50000;
  config.freq_hi = 55000;
  config.freq_sample = 200000;
  config.freq_bit = 10;
  config.freq_sync_bit = 2;
  config.sdft_N = 500;
  config.sdft_r = 0.99;
  config.max_payload_length = 16;

  config.fast_bit_coeff = 0.01;
  config.slow_bit_coeff = 0.001;
  config.edge_threshold = -0.5;

  tx::setup(&config);
  rx::setup(&config);
}

void loop()
{
  // while (!Serial)
  //   ;

  // char line[16];

  // if (!Serial.readBytesUntil('\r', line, sizeof(line)))
  //   return;

  // while (Serial.available() > 0)
  // {
  //   char t = Serial.read();
  // }

  tx_frame.payload_length = 1;
  ++tx_frame.payload[0];
  // memcpy(tx_frame.payload, &line[1], sizeof(line) - 1);

  tx_frame.update_checksum();

  tx::transmit(&tx_frame);

  Serial.println("done tx, start receive");

  bool received = rx::receive(&rx_frame, 30 * ONE_S_IN_NS);

  if (rx_frame.check_checksum() && received)
  {
    Serial.println("Received message!");
    Serial.write(rx_frame.payload_length);
    Serial.write(rx_frame.payload, rx_frame.payload_length);
    Serial.print('\r\n');
  }
}
