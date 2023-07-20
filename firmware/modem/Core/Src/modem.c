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

status_t demodulator_init(gsfk_demod_t* m, const demodulator_config_t* c)
{
	m->c = *c;
	m->raw_buf = malloc(sizeof(d_raw_t) * c->chip_buf_size * 2);

	return MDM_OK;
}

status_t demodulator_start(gsfk_demod_t* modem)
{
	HAL_ADC_Start_DMA(modem->c.hadc, modem->raw_buf, modem->c.chip_buf_size);
}

void modem_adc_conv_half_cplt_cb(gsfk_demod_t* m)
{
	m->writing_status = WRITING_BACK;
}

void modem_adc_conv_cplt_cb(gsfk_demod_t* m)
{
	m->writing_status = WRITING_FRONT;
}

status_t modem_sdft(gsfk_demod_t* m)
{
	demodulator_config_t* c = &m->c;
	d_raw_t* curr_write_addr = (d_raw_t*)c->hadc->DMA_Handle->Instance->M0AR;

	// Check margin
	bool ok;
	d_raw_t* half = m->raw_buf + sizeof(d_raw_t) * c->chip_buf_size / 2;
	d_raw_t* end = m->raw_buf + sizeof(d_raw_t) * c->chip_buf_size;
	if (m->writing_status == WRITING_FRONT)
	{
		ok = (half <= curr_write_addr && curr_write_addr < (end - c->chip_buf_margin));
	}
	else
	{
		ok = (m->raw_buf + c->chip_buf_margin <= curr_write_addr && curr_write_addr < end);
	}
	if (!ok) return MDM_ERROR;

	size_t transform_start, transform_end;
	if (m->writing_status == WRITING_BACK)
	{
		transform_start = 0;
		transform_end = c->chip_buf_size / 2;
	}
	else
	{
		transform_start = c->chip_buf_size / 2;
		transform_end = c->chip_buf_size;
	}

	for (size_t i = transform_start; i < transform_end; i++)
	{
		float sample = ((float)m->raw_buf[i]) / c->max_raw;
		float sample_N = ((float)m->raw_buf[(i - c->chip_sdft_N) % c->chip_buf_size]) / c->max_raw;
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

	}
}