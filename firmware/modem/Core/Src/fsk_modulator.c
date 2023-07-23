//
// Created by Gleb Ryabtsev on 7/22/23.
//

#include "fsk_modulator.h"
#include "math.h"

/* FUNCTION PROTOTYPES */

uint32_t min(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

float gaussian_integral(float x, float s) {
    return x
          - powf(x, 3) / (6 * powf(s, 2))
          + powf(x, 5) / (40 * powf(s, 4))
          - powf(x, 7) / (336 * powf(s, 6));
}

status_t fsk_mod_init(fsk_modulator_t *mod, fsk_modulator_config_t *config) {
    mod->c = *config;

    /* Generate LUTs */
    // normalize and map to the half-periods
    float k = 1.0f / gaussian_integral(1.0f, config->sigma);
    uint32_t hp_lo = SystemCoreClock / (config->modem_config.freq_lo * 2);
    uint32_t hp_hi = SystemCoreClock / (config->modem_config.freq_hi * 2);
    float m = ((float)hp_hi - (float)hp_lo) / 2.0f;
    float n = ((float) (hp_lo + hp_hi)) / 2.0f;
    for (int i = 0; i < FREQ_LUT_SIZE; i++) {
        float x = -1.0f + 2.0f * ((float) i) / FREQ_LUT_SIZE;
        float g = k * gaussian_integral(x, config->sigma);
        mod->period_lut[i] = (uint32_t) (m*g + n);
    }

    mod->bit_period = SystemCoreClock / config->modem_config.chip_rate;
    mod->lut_time_coeff = (float) FREQ_LUT_SIZE / (float) mod->bit_period;

    mod->transmitting = false;
    mod->buf = NULL;
    mod->buf_size = 0;
    mod->buf_time_end = 0;
    mod->buf_time = 0;

    return MDM_OK;
}

status_t fsk_mod_transmit(fsk_modulator_t *mod, m_word_t *buf, size_t size) {
    mod->buf = buf;
    mod->buf_size = size;
    mod->prev_bit = buf[0] & 0x1;
    mod->curr_bit = buf[0] & 0x1;
    mod->buf_time = 0;
    mod->buf_time_end = mod->bit_period * (size * 8);
    mod->transmitting = true;
    mod->next_bit_index = 1;

    mod->c.pwm_tim->Instance->ARR = 0x100;

    HAL_GPIO_WritePin(TX_EN_GPIO_Port, TX_EN_Pin, 1);
    HAL_TIM_Base_Start_IT(mod->c.pwm_tim);
    return MDM_OK;
}

bool fsk_mod_busy(fsk_modulator_t *mod) {
    return mod->transmitting;
}

void fsk_mod_tim_it(fsk_modulator_t *mod) {
    if (mod->buf_time >= mod->buf_time_end) {
        HAL_TIM_Base_Stop_IT(mod->c.pwm_tim);
        HAL_GPIO_WritePin(TX_GPIO_Port, TX_Pin, 0);
        HAL_GPIO_WritePin(TX_EN_GPIO_Port, TX_EN_Pin, 0);
        mod->transmitting = false;
        return;
    }

    HAL_GPIO_TogglePin(TX_GPIO_Port, TX_Pin);

    size_t next_bit_i = mod->buf_time / mod->bit_period + 1; // can go 1 bit past the buffer
    if (next_bit_i != mod->next_bit_index) {
        mod->next_bit_index = next_bit_i;
        mod->prev_bit = mod->curr_bit;
        mod->curr_bit = mod->next_bit;

        next_bit_i = min(next_bit_i, mod->buf_size * 8 - 1);
        size_t next_byte_i = next_bit_i / 8;
        uint8_t msk = 0x1 << (next_bit_i % 8);
        mod->next_bit = mod->buf[next_byte_i] & msk;
    }

    uint32_t T = (uint32_t) ((float) (mod->buf_time % mod->bit_period) * mod->lut_time_coeff); // TODO: replace with int div
    uint32_t period;
    if (T < FREQ_LUT_SIZE / 2) {
        if (mod->prev_bit && !mod->curr_bit) {
            period = mod->period_lut[FREQ_LUT_SIZE / 2 - T - 1];
        } else if (!mod->prev_bit && mod->curr_bit){
            period = mod->period_lut[FREQ_LUT_SIZE / 2 + T];
        } else {
            size_t idx = mod->curr_bit ? FREQ_LUT_SIZE - 1 : 0;
            period = mod->period_lut[idx];
        }
    } else {
        if (!mod->curr_bit && mod->next_bit) {
            period = mod->period_lut[T - FREQ_LUT_SIZE / 2];
        } else if (mod->curr_bit & !mod->next_bit) {
            period = mod->period_lut[FREQ_LUT_SIZE * 3 / 2 - 1 - T];
        } else {
            size_t idx = mod->curr_bit ? FREQ_LUT_SIZE - 1 : 0;
            period = mod->period_lut[idx];
        }
    }
    mod->c.pwm_tim->Instance->ARR = period;
    mod->buf_time += period;

}
