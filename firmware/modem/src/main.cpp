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
TeensyTimerTool::PeriodicTimer demodTimer(TeensyTimerTool::GPT2);

void demod_adc_it();

modem_config_t modemConfig{
    .freq_lo = 50000,
    .freq_hi = 60000,
    .chip_rate = 500,
};

FSKModulator mod{&modemConfig, &fskTimer};
FSKDemodulator demod{&modemConfig, &demodTimer, RAW_BUF_SIZE, adc_raw_buf, sdft_buf_1, sdft_buf_2, demod_adc_it};

FSKModulator::m_word_t buf[] = {0b01010101, };

void setup() {
    pinMode(2, OUTPUT); // Initialize LED
    pinMode(3, OUTPUT);
    pinMode(4, OUTPUT);

    Serial.begin(115200);
    Serial.println("Beginning setup...");
    mod.setSigma(0.6);
    mod.init();
    Serial.println("FKS Mod init OK");
    demod.init();
    Serial.println("Demod init OK");
    demod.start();
    Serial.println("Setup completed!");
}

void loop() {
    delay(10);
//    while(mod.fsk_mod_busy()) {
////        Serial.println("Waiting");
////        delay(10);
//    }
//
//    mod.fsk_mod_transmit(buf, 1);
//    delay(20);
}

void demod_adc_it() {
    demod.handle_sample();
}
