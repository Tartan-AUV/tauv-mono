//
// Created by Gleb Ryabtsev on 7/22/23.
//

#ifndef MODEM_FSK_MODULATOR_H
#define MODEM_FSK_MODULATOR_H

#include "main.h"
#include "stm32f4xx_hal.h"
#include "modem.h"

#define FREQ_LUT_SIZE 20

typedef struct {
    modem_config_t modem_config;
    TIM_HandleTypeDef *pwm_tim;
    float sigma;
} fsk_modulator_config_t;


typedef struct {
    fsk_modulator_config_t c;
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
} fsk_modulator_t;

typedef uint8_t m_word_t;

status_t fsk_mod_init(fsk_modulator_t *mod, fsk_modulator_config_t *config);

status_t fsk_mod_transmit(fsk_modulator_t *mod, m_word_t *buf, size_t size);

bool fsk_mod_busy(fsk_modulator_t *mod);

void fsk_mod_tim_it(fsk_modulator_t *mod);
#endif //MODEM_FSK_MODULATOR_H
