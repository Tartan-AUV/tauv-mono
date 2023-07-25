//
// Created by Gleb Ryabtsev on 7/25/23.
//

#include <Arduino.h>

#include "main.h"
#include "fsk_modulator.h"

d_raw_t adc_raw_buf[RAW_BUF_SIZE];
d_sdft_t sdft_buf_1[SDFT_BUF_SIZE];
d_sdft_t sdft_buf_2[SDFT_BUF_SIZE];

TeensyTimerTool::PeriodicTimer fskTimer(TeensyTimerTool::GPT1);

modem_config_t modemConfig{
    .freq_lo = 50000,
    .freq_hi = 60000,
    .chip_rate = 500,
};

FSKModulator mod{&modemConfig, &fskTimer};

FSKModulator::m_word_t buf[] = {0b01010101, };

void setup() {
    mod.setSigma(0.6);
    mod.init();
    Serial.begin(115000);
    Serial.println("Setup completed!");
}

void loop() {
    while(mod.fsk_mod_busy()) {
//        Serial.println("Waiting");
//        delay(10);
    }

    mod.fsk_mod_transmit(buf, 1);
    delay(20);
}
