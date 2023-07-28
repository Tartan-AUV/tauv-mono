////
//// Created by Gleb Ryabtsev on 7/19/23.
////
//
#pragma once

#include <math.h>
#include <complex.h>

#include <TeensyTimerTool.h>
#include <ADC.h>
#include "main.h"

//
typedef uint32_t d_raw_t;
typedef int8_t d_sdft_t;
typedef void (*sdft_buf_cplt_fn)();
typedef void (*adc_it_fn)();

//void aintdoingshit() {}

class FSKDemodulator {
public:
    FSKDemodulator(modem_config_t *modemConfig, TeensyTimerTool::PeriodicTimer *sampleTimer,
                   unsigned int rawSize, d_raw_t *rawBuf, d_sdft_t *dstBuf1, d_sdft_t *dstBuf2,
                   adc_it_fn adc_it, size_t dstSize);

    status_t init();

    void handle_sample();// __attribute__((section (".ccmram")));

    status_t start(sdft_buf_cplt_fn cplt1, sdft_buf_cplt_fn cplt2);

//    status_t demodulator_deinit();
private:
    float normalize_sample(d_raw_t sample);


    TeensyTimerTool::PeriodicTimer *sampleTimer;

    modem_config_t *modem_config;
    size_t raw_size;
    d_raw_t *raw_buf;
    d_sdft_t *dst_buf1;
    d_sdft_t *dst_buf2;
    size_t undersampling_ratio = SDFT_UNDERSAMPLING_RATIO;

    float sdft_r = 0.99;

    float max_raw = 255.0;

    adc_it_fn adc_it;

    uint32_t sample_period = 5000; // todo: add setter

    static constexpr float pi = 3.14159f;

    sdft_buf_cplt_fn cplt1 = demod_nop;
    sdft_buf_cplt_fn cplt2 = demod_nop;

    /*v*/size_t i, dst_i;
    size_t dst_size;
    /*v*/size_t N;
    /*v*/d_sdft_t *curr_dst_buf;


    /*v*/float k_lo, k_hi;
    /*v*/float coeff_w[3];
    /*v*/float coeff_a;
    std::complex<float> coeff_b_lo[3];
    std::complex<float> coeff_b_hi[3];

    /*v*/std::complex<float> s_lo_w[3];
    /*v*/std::complex<float> s_hi_w[3];

    /*v*/std::complex<float> s_lo;
    /*v*/std::complex<float> s_hi;
//
    /*v*/float mag_lo;
    /*v*/float mag_hi;

    static void demod_nop() {};
};
