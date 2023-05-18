#include <Arduino.h>
#include <TeensyTimerTool.h>

#include "config.h"
#include "tx.h"
#include "rx.h"

#define RX_MODE

// Serial commands
// ros -> teensy
#define SET_MODE 0x00
#define SET_PARAM 0x01
#define MASTER_REQ 0x02
#define SLAVE_RESP 0x03

// teensy -> ros
#define SLAVE_REQ 0x04
#define MASTER_RESP 0x05

// Modes
#define MODE_MASTER 0x00
#define MODE_SLAVE 0x01

uint8_t mode = MODE_MASTER;

static Config config;

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

void set_param(uint8_t param, uint32_t val) {
  switch (param) {
    case FREQ_LO: {
      config.freq_lo = val;
      break;
    }
    case FREQ_HI: {
      config.freq_hi = val;
      break;
    }
    case FREQ_SAMPLE: {
      config.freq_sample = val;
      break;
    }
    case FREQ_BIT: {
      config.freq_bit = val;
      break;
    }
    case FREQ_SYNC_BIT: {
      config.freq_sync_bit = val;
      break;
    }
    case SDFT_N: {
      config.sdft_N = val;
      break;
    }
    case SDFT_R: {
      config.sdft_r = *(float*)&val;
      break;
    }
    case MAX_PAYLOAD_LENGTH: {
      config.max_payload_length = val;
      break;
    }
    case FILTER_COEFF: {
      config.slow_bit_coeff = *(float*)&val;
      break;
    }
    case EDGE_THRESHOLD: {
      config.edge_threshold = *(float*)&val;
      break;
    }
    case RECEIVE_TIMEOUT: {
      config.receive_timeout = std::chrono::milliseconds(val);
      break;
    }
    default: {
      break;
    }
  }
  tx::setup(&config);
  rx::setup(&config);
}

void loop_master() {
  while (!Serial.available())
    ;

  uint8_t cmd = Serial.read();
  if (cmd == SET_MODE) {
    uint8_t m = Serial.read();
    if (m == MODE_MASTER || m == MODE_SLAVE) {
      mode = m;
    }
  } else if (cmd == SET_PARAM) {
    uint8_t param = Serial.read();
    uint32_t val;
    Serial.readBytes((char*)&val, 4);
    set_param(param, val);
  } else if (cmd == MASTER_REQ) {

    Frame tx_frame;
    tx_frame.payload_length = Serial.read();
    Serial.readBytes((char*)tx_frame.payload, tx_frame.payload_length);
    tx::transmit(&tx_frame);

    Frame rx_frame;
    bool received = rx::receive(&rx_frame, config.receive_timeout);
    if (received) {
      Serial.write(MASTER_RESP);
      Serial.write(rx_frame.payload_length);
      Serial.write(rx_frame.payload, rx_frame.payload_length);
    }
  }
}

void loop_slave() {
  if (Serial.available()) {
    uint8_t cmd = Serial.read();
    if (cmd == SET_MODE) {
      uint8_t m = Serial.read();
      if (m == MODE_MASTER || m == MODE_SLAVE) {
        mode = m;
      }
    } else if (cmd == SET_PARAM) {
      uint8_t param = Serial.read();
      uint32_t val;
      Serial.readBytes((char*)&val, 4);
      set_param(param, val);
    }
  }

  Frame rx_frame;
  bool success = rx::receive(&rx_frame, 3 * ONE_S_IN_NS * 2);
  if (!success) {
    return;
  }

  Serial.write(SLAVE_REQ);
  Serial.write(rx_frame.payload_length);
  Serial.write(rx_frame.payload, rx_frame.payload_length);

  while (!Serial.available())
    ;
  
  uint8_t cmd = Serial.peek();
  if (cmd == MASTER_RESP) {
    Serial.read();
    Frame tx_frame;
    tx_frame.payload_length = Serial.read();
    Serial.readBytes((char*)tx_frame.payload, tx_frame.payload_length);
    tx::transmit(&tx_frame);
  } else {
    return; // go back to the beginning of the loop
  }
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
  if (mode == MODE_MASTER) {
    loop_master();
  } else if (mode == MODE_SLAVE) {
    loop_slave();
  }
}