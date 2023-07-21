//
// Created by Gleb Ryabtsev on 7/19/23.
//

#include <malloc.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

#include "modem.h"
#include "stm32f4xx_hal.h"
#include "main.h"

status_t demodulator_init(demod_t *m, const demodulator_config_t *c) {
    m->c = *c;
    m->raw_buf = malloc(sizeof(d_raw_t) * c->chip_buf_size * 2);
    m->sdft_buf = malloc(sizeof(d_sdft_t) * c->chip_buf_size * 2);
    if (!m->raw_buf || !m->sdft_buf) {
        return MDM_ERROR;
    }
    int freq_sample = 956000; // todo
    float PI = 3.14159f;
    float N = m->c.chip_sdft_N;
    m->k_lo = ((float) m->c.modem_config.freq_lo * (float) N) / ((float) freq_sample);
    m->k_hi = ((float) m->c.modem_config.freq_hi * (float) N) / ((float) freq_sample);

    m->coeff_w[0] = -0.25f;
    m->coeff_w[1] = 0.5f;
    m->coeff_w[2] = -0.25f;

    float r = m->c.sdft_r;
    m->coeff_a = -pow(r, (float) N);

    m->coeff_b_lo[0] = r * cexp((2.i * PI * (m->k_lo - 1.)) / (float) N);
    m->coeff_b_lo[1] = r * cexp((2.i * PI * m->k_lo) / (float) N);
    m->coeff_b_lo[2] = r * cexp((2.i * PI * (m->k_lo + 1.)) / (float) N);
    m->coeff_b_hi[0] = r * cexp((2.i * PI * (m->k_hi - 1.)) / (float) N);
    m->coeff_b_hi[1] = r * cexp((2.i * PI * m->k_hi) / (float) N);
    m->coeff_b_hi[2] = r * cexp((2.i * PI * (m->k_hi + 1.)) / (float) N);

    return MDM_OK;
}

status_t demodulator_start(demod_t *demod) {
    HAL_ADC_Start_DMA(demod->c.hadc, demod->raw_buf, demod->c.chip_buf_size);
}

void demod_adc_conv_half_cplt_cb(demod_t *m) {
    m->writing_status = WRITING_BACK;
    m->raw_buf_rdy = true;
}

void demod_adc_conv_cplt_cb(demod_t *m) {
    m->writing_status = WRITING_FRONT;
    m->raw_buf_rdy = true;
}

status_t modem_sdft(demod_t *m) {
    demodulator_config_t *c = &m->c;
    d_raw_t *curr_write_addr = (d_raw_t *) c->hadc->DMA_Handle->Instance->M0AR;

    // Check margin
    bool ok;
    d_raw_t *half = m->raw_buf + sizeof(d_raw_t) * c->chip_buf_size / 2;
    d_raw_t *end = m->raw_buf + sizeof(d_raw_t) * c->chip_buf_size;
    if (m->writing_status == WRITING_FRONT) {
        ok = (half <= curr_write_addr && curr_write_addr < (end - c->chip_buf_margin));
    } else {
        ok = (m->raw_buf + c->chip_buf_margin <= curr_write_addr && curr_write_addr < end);
    }
    if (!ok) return MDM_ERROR;

    size_t transform_start, transform_end;
    if (m->writing_status == WRITING_BACK) {
        transform_start = 0;
        transform_end = c->chip_buf_size / 2;
    } else {
        transform_start = c->chip_buf_size / 2;
        transform_end = c->chip_buf_size;
    }

    for (size_t i = transform_start; i < transform_end; i++) {
        float sample = ((float) m->raw_buf[i]) / c->max_raw;
        float sample_N = ((float) m->raw_buf[(i - c->chip_sdft_N) % c->chip_buf_size]) / c->max_raw;
        // potential optimization through storing a parallel float list

        float a = sample - m->coeff_a * sample_N;

        m->s_lo_w[0] = a + m->coeff_b_lo[0] * m->s_lo_w[0];
        m->s_lo_w[1] = a + m->coeff_b_lo[1] * m->s_lo_w[1];
        m->s_lo_w[2] = a + m->coeff_b_lo[2] * m->s_lo_w[2];

        m->s_hi_w[0] = a + m->coeff_b_hi[0] * m->s_hi_w[0];
        m->s_hi_w[1] = a + m->coeff_b_hi[1] * m->s_hi_w[1];
        m->s_hi_w[2] = a + m->coeff_b_hi[2] * m->s_hi_w[2];

        m->s_lo = m->coeff_w[0] * m->s_lo_w[0] + m->coeff_w[1] * m->s_lo_w[1] + m->coeff_w[2] * m->s_lo_w[2];
        m->s_hi = m->coeff_w[0] * m->s_hi_w[0] + m->coeff_w[1] * m->s_hi_w[1] + m->coeff_w[2] * m->s_hi_w[2];

        m->mag_lo = cabs(m->s_lo);
        m->mag_hi = cabs(m->s_hi);

        m->sdft_buf[i] = m->mag_hi - m->mag_lo;
    }
}

status_t demodulator_update(demod_t *demod) {
    if (!demod->raw_buf_rdy) return MDM_OK;

    modem_sdft(demod);

    return MDM_OK;
}
//status_t seq_decode()