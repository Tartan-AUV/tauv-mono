////
//// Created by Gleb Ryabtsev on 7/19/23.
////
//
#pragma once

#include <Arduino.h>
#undef abs

#include <math.h>
#include <complex.h>

#include <TeensyTimerTool.h>
#include <ADC.h>

typedef enum {
    MDM_OK,
    MDM_ERROR
} status_t;
//
typedef uint8_t d_raw_t;
typedef _Complex float d_sdft_t;
typedef void (*sdft_buf_cplt_fn)();
typedef void (*adc_it_fn)();
//
typedef struct {
    int freq_lo;
    int freq_hi;
    int chip_rate;
} modem_config_t;

//void aintdoingshit() {}

class FSKDemodulator {
public:
    FSKDemodulator(modem_config_t *modemConfig, TeensyTimerTool::PeriodicTimer *sampleTimer,
                   unsigned int rawSize, unsigned char *rawBuf, d_sdft_t *dstBuf1, d_sdft_t *dstBuf2,
                   adc_it_fn adc_it);

    status_t init();

    void handle_sample();// __attribute__((section (".ccmram")));

    status_t start();

//    status_t demodulator_deinit();
private:
    float normalize_sample(d_raw_t sample);

    ADC adc;

    TeensyTimerTool::PeriodicTimer *sampleTimer;

    modem_config_t *modem_config;
    size_t raw_size;
    d_raw_t *raw_buf;
    d_sdft_t *dst_buf1;
    d_sdft_t *dst_buf2;
    size_t undersampling_ratio = 8;

    _Complex float sdft_r = 0.99;

    _Complex float max_raw = 255.0;

    adc_it_fn adc_it;

    uint32_t sample_period = 5000; // todo: add setter

    static constexpr _Complex float pi = 3.14159f;

    sdft_buf_cplt_fn cplt1 = nullptr;
    sdft_buf_cplt_fn cplt2 = nullptr;

    volatile size_t i, dst_i;
    size_t dst_size;
    volatile size_t N;
    volatile d_sdft_t *curr_dst_buf;


    volatile _Complex float k_lo, k_hi;
    volatile _Complex float coeff_w[3];
    volatile _Complex float coeff_a;
    volatile _Complex float coeff_b_lo[3];
    volatile _Complex float coeff_b_hi[3];

    volatile _Complex float s_lo_w[3];
    volatile _Complex float s_hi_w[3];

    volatile _Complex float s_lo;
    volatile _Complex float s_hi;
//
    volatile _Complex float mag_lo;
    volatile _Complex float mag_hi;
};