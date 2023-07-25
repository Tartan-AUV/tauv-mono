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

typedef uint8_t d_raw_t;
typedef float d_sdft_t;
typedef void (*sdft_buf_cplt_fn)(void);

typedef struct {
    int freq_lo;
    int freq_hi;
    int chip_rate;
} modem_config_t;

typedef struct {
    modem_config_t modem_config;
    size_t raw_size;

    d_raw_t *raw_buf;
    d_sdft_t *dst_buf1;
    d_sdft_t *dst_buf2;
    size_t undersampling_ratio;

    float sdft_r;

    float max_raw;

    ADC_HandleTypeDef *hadc;
    TIM_HandleTypeDef *sample_tim;

    sdft_buf_cplt_fn cplt1;
    sdft_buf_cplt_fn cplt2;

} demodulator_config_t;

typedef struct {
    demodulator_config_t c;

    uint32_t adc_data;

    volatile size_t i, dst_i;
    size_t dst_size;
    volatile size_t N;
    volatile d_sdft_t *curr_dst_buf;


    volatile float k_lo, k_hi;
    volatile float coeff_w[3];
    volatile float coeff_a;
    __complex__ float coeff_b_lo[3];
    __complex__ float coeff_b_hi[3];

    volatile __complex__ float s_lo_w[3];
    volatile __complex__ float s_hi_w[3];

    volatile __complex__ float s_lo;
    volatile __complex__ float s_hi;

    volatile float mag_lo;
    volatile float mag_hi;

//    volatile float slow_bit_avg;
//    volatile float fast_bit_avg;
} demod_t;

status_t demodulator_init(demod_t *m, const demodulator_config_t *c);


void demod_sample_it(demod_t *m);// __attribute__((section (".ccmram")));

status_t demodulator_start(demod_t *demod);

status_t demodulator_deinit(demod_t *d);

#endif //MODEM_CORE_INC_MODEM_H_
