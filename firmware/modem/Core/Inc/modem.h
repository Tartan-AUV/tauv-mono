//
// Created by Gleb Ryabtsev on 7/19/23.
//

#ifndef MODEM_CORE_INC_MODEM_H_

#define MODEM_CORE_INC_MODEM_H_

#include <stdbool.h>
#include "stm32f4xx_hal.h"

typedef enum
{
	MDM_OK,
	MDM_ERROR
} status_t;

typedef uint32_t d_raw_t;
typedef float d_sdft_t;

typedef struct
{
	int freq_lo;
	int freq_hi;
	int chip_rate;
} modem_config_t;

typedef struct
{
	modem_config_t modem_config;
	size_t chip_buf_size;

	size_t chip_buf_margin;
	size_t chip_sdft_N;

	float max_raw;

	ADC_HandleTypeDef* hadc;
} demodulator_config_t;

typedef struct
{
	demodulator_config_t c;

	d_raw_t* raw_buf;
	d_sdft_t *sdft_buf;

	uint8_t *b_seq;

	enum
	{
		WRITING_FRONT,
		WRITING_BACK
	} writing_status;

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
} gsfk_demod_t;

status_t demodulator_init(gsfk_demod_t* m, const demodulator_config_t* c);

status_t demodulator_update(gsfk_demod_t* m);

status_t demodulator_start(gsfk_demod_t* modem);

void modem_adc_conv_half_cplt_cb(gsfk_demod_t* m);

void modem_adc_conv_cplt_cb(gsfk_demod_t* m);

status_t demodulator_deinit(gsfk_demod_t* d);

#endif //MODEM_CORE_INC_MODEM_H_
