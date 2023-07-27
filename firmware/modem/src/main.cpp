//
// Created by Gleb Ryabtsev on 7/25/23.
//

#include <Arduino.h>

#include "main.h"
#include "fsk_modulator.h"

#define TX_DBG

d_raw_t adc_raw_buf[RAW_BUF_SIZE];
d_sdft_t sdft_buf_1[SDFT_BUF_SIZE];
d_sdft_t sdft_buf_2[SDFT_BUF_SIZE];

TeensyTimerTool::PeriodicTimer fskTimer(TeensyTimerTool::GPT1);
TeensyTimerTool::PeriodicTimer demodTimer(TeensyTimerTool::GPT2);

void demod_adc_it();

modem_config_t modemConfig{
    .freq_lo = 49000,
    .freq_hi = 54000, // 56k on rx, 60k on tx
    .chip_rate = 500,
};

FSKModulator mod{&modemConfig, &fskTimer};
FSKDemodulator demod{&modemConfig, &demodTimer, RAW_BUF_SIZE, adc_raw_buf, sdft_buf_1, sdft_buf_2, demod_adc_it};

FSKModulator::m_word_t buf[] = {0x55};

void setup() {
    mod.setSigma(0.5);
    mod.init();
    demod.init();
#ifdef RX_DBG
    demod.start();
#endif
    Serial.begin(115000);
    Serial.println("Setup completed!");

    pinMode(PIN_DBG_1, OUTPUT);
    pinMode(PIN_DBG_2, OUTPUT);
}

void loop() {
#ifdef TX_DBG
    while(mod.fsk_mod_busy()) {
    }
    delay(20);
    mod.fsk_mod_transmit(buf, 1);
//    Serial.println("Running");
#endif
#ifdef RX_DBG
#endif
}

void demod_adc_it() {
    demod.handle_sample();
}
