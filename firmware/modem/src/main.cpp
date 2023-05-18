#include <Arduino.h>
#include <TeensyTimerTool.h>

#include "config.h"
#include "tx.h"
#include "rx.h"

#define RX_MODE

static Config config;
static Frame tx_frame;
static Frame rx_frame;

void printBuffer(uint8_t* buffer, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (buffer[i] < 0x10) {
      Serial.print("0");
    }
    Serial.print(buffer[i], HEX);
    Serial.print(" ");
  }
  Serial.println();
}

void setup()
{
  Serial.begin(115200);
  Serial.setTimeout(100);

  config.freq_lo = 50000;
  config.freq_hi = 55000;
  config.freq_sample = 200000;
  config.freq_bit = 100;
  config.freq_sync_bit = 20;
  config.sdft_N = 500;
  config.sdft_r = 0.99;
  config.max_payload_length = 16;

  config.fast_bit_coeff = 0.0; // not used
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

  #ifdef TX_MODE
  tx_frame.payload_length = 4;
  // tx_frame.payload[0] = 0xFF;
  ++tx_frame.payload[0];
  tx_frame.payload[1] = 0x42;
  tx_frame.payload[2] = 0x42;
  tx_frame.payload[3] = 0x43;

  tx::transmit(&tx_frame);

  delay(100);
  #endif

  #ifdef RX_MODE
  // Serial.println("Receiving");
  volatile bool received = rx::receive(&rx_frame, 3 * ONE_S_IN_NS);
  if (received)
  {
    // Serial.println("Received message!");
    Serial.print(rx_frame.payload_length);
    Serial.print("   |   ");
    printBuffer(rx_frame.payload, rx_frame.payload_length);
  } else {
    Serial.println("Error");
  }
  #endif
}
