//
// Created by Gleb Ryabtsev on 7/22/23.
//

#ifndef MODEM_FSK_MODULATOR_H
#define MODEM_FSK_MODULATOR_H


#include "modem.h"

#define FREQ_LUT_SIZE 20

class FSKModulator {
public:
    FSKModulator(modem_config_t *modemConfig, TeensyTimerTool::PeriodicTimer *t);

    void setSigma(float sigma);

    typedef uint8_t m_word_t;

    status_t init();

    status_t fsk_mod_transmit(m_word_t *buf, size_t size);

    bool fsk_mod_busy();

    void fsk_mod_tim_it();

private:
    // Config
    modem_config_t *modem_config;
    TeensyTimerTool::PeriodicTimer *t;
    float sigma;

    // State / running
    uint32_t bit_period;
    uint32_t period_lut[FREQ_LUT_SIZE];
    float lut_time_coeff;

    uint8_t *buf;
    size_t buf_size;
    bool transmitting;

    uint32_t buf_time_end;
    uint32_t buf_time;

    bool prev_bit, curr_bit, next_bit;
    size_t next_bit_index;

    uint32_t period_hi, period_lo;

    void fsk_mod_transmit_isr_no_g();
};

#endif //MODEM_FSK_MODULATOR_H
