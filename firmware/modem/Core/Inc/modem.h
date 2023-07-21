//
// Created by Gleb Ryabtsev on 7/19/23.
//

#ifndef MODEM_CORE_INC_MODEM_H_

#define MODEM_CORE_INC_MODEM_H_

#include <stdbool.h>
#include "stm32f4xx_hal.h"

typedef enum {
    MDM_OK,
    MDM_ERROR
} status_t;

typedef uint32_t d_raw_t;
typedef float d_sdft_t;

typedef struct {
    int freq_lo;
    int freq_hi;
    int chip_rate;
} modem_config_t;

typedef struct {
    modem_config_t modem_config;
    size_t chip_buf_size;

    size_t chip_buf_margin;
    size_t chip_sdft_N;

    d_raw_t *combined_raw_buf;

    float sdft_r;

    float max_raw;

    ADC_HandleTypeDef *hadc;

} demodulator_config_t;

typedef struct {
    demodulator_config_t c;

//    d_raw_t *raw_buf;
//    d_sdft_t *sdft_buf;

    d_raw_t *raw_writing_buf, *raw_last_buf, *raw_prev_buf;

//    uint8_t *b_seq;

    bool raw_buf_rdy;

    float k_lo, k_hi;
    float coeff_w[3];
    float coeff_a;
    __complex__ float coeff_b_lo[3];
    __complex__ float coeff_b_hi[3];

    volatile __complex__ float s_lo_w[3];
    volatile __complex__ float s_hi_w[3];

    volatile __complex__ float s_lo;
    volatile __complex__ float s_hi;

    volatile float mag_lo;
    volatile float mag_hi;

    volatile float slow_bit_avg;
    volatile float fast_bit_avg;
} demod_t;

status_t demodulator_init(demod_t *m, const demodulator_config_t *c);

status_t demod_sdft(demod_t *m, d_sdft_t *dst, size_t dst_size);

status_t demodulator_start(demod_t *demod);

void demod_adc_dma_m0_cplt_it(demod_t *m);

void demod_adc_dma_m1_cplt_it(demod_t *m);

status_t demodulator_deinit(demod_t *d);

#endif //MODEM_CORE_INC_MODEM_H_
