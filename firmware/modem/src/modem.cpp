////
//// Created by Gleb Ryabtsev on 7/19/23.
////
//
//#include <math.h>
//#include <complex.h>
//
#include "modem.h"
#include "main.h"

static ADC adc;

float mi, ma;


using namespace std::complex_literals;

status_t FSKDemodulator::init() {
    using namespace std;
    int freq_sample = 100000; // todo
    i = 0;
    curr_dst_buf = dst_buf1;
    dst_size = raw_size / undersampling_ratio;
    N = raw_size + 1;
    dst_i = 0;

    float pi = 3.14159f;
    float N = (float) raw_size;
    k_lo = ((float) modem_config->freq_lo * (float) N) / ((float) freq_sample);
    k_hi = ((float) modem_config->freq_hi * (float) N) / ((float) freq_sample);

    coeff_w[0] = -0.25f;
    coeff_w[1] = 0.5f;
    coeff_w[2] = -0.25f;

    float r = sdft_r;
    coeff_a = -pow(r, (float) N);

    coeff_b_lo[0] = r * exp((2.if * pi * (k_lo - 1.0f)) / (float) N);
    coeff_b_lo[1] = r * exp((2.if * pi * k_lo) / (float) N);
    coeff_b_lo[2] = r * exp((2.if * pi * (k_lo + 1.0f)) / (float) N);
    coeff_b_hi[0] = r * exp((2.if * pi * (k_hi - 1.0f)) / (float) N);
    coeff_b_hi[1] = r * exp((2.if * pi * k_hi) / (float) N);
    coeff_b_hi[2] = r * exp((2.if * pi * (k_hi + 1.0f)) / (float) N);
    pinMode(PIN_RX, INPUT_DISABLE);
    adc.adc0->setAveraging(1);
    adc.adc0->setResolution(8);
    adc.adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
    adc.adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);
    adc.adc0->enableInterrupts(adc_it);

//    adc.adc0->startContinuous(PIN_RX);
    adc.adc0->startSingleRead(PIN_RX);

    return MDM_OK;
}

status_t FSKDemodulator::start()  {
    sampleTimer->begin([this] {
        adc.adc0->startSingleRead(PIN_RX);
        },
                       std::chrono::nanoseconds (sample_period), true);

    return MDM_OK;
}


float FSKDemodulator::normalize_sample(d_raw_t sample) {
    return ((float) sample) * (1.0 / 256.0);
}

void FSKDemodulator::handle_sample() {
    digitalWriteFast(PIN_DBG_2, 1);
    auto val = adc.adc0->readSingle();

    float sample = normalize_sample(val);
    float sample_N = normalize_sample(raw_buf[i]);

    float a = sample - coeff_a * sample_N;

    s_lo_w[0] = a + coeff_b_lo[0] * s_lo_w[0];
    s_lo_w[1] = a + coeff_b_lo[1] * s_lo_w[1];
    s_lo_w[2] = a + coeff_b_lo[2] * s_lo_w[2];

    s_hi_w[0] = a + coeff_b_hi[0] * s_hi_w[0];
    s_hi_w[1] = a + coeff_b_hi[1] * s_hi_w[1];
    s_hi_w[2] = a + coeff_b_hi[2] * s_hi_w[2];

    s_lo = coeff_w[0] * s_lo_w[0] + coeff_w[1] * s_lo_w[1] + coeff_w[2] * s_lo_w[2];
    s_hi = coeff_w[0] * s_hi_w[0] + coeff_w[1] * s_hi_w[1] + coeff_w[2] * s_hi_w[2];

    mag_lo = std::abs(s_lo);
    mag_hi = std::abs(s_hi);

    if(i % undersampling_ratio) {
        curr_dst_buf[dst_i] = mag_hi - mag_lo;
        dst_i++;
        if(dst_i > dst_size) {
            if (curr_dst_buf == dst_buf1) {
//                Serial.println();
                Serial.println(mag_hi);
                Serial.println(mag_lo);
                Serial.println();
                (*cplt1)();
                curr_dst_buf = dst_buf2;
            }
            else {
                (*cplt2)();
                curr_dst_buf = dst_buf1;
            }
            i = 0;
            dst_i = 0;
        }
    }
    raw_buf[i] = val;
    i = (i + 1) % raw_size;

    if (mag_hi > ma) {
        ma = mag_hi;
    }
    if (mag_lo < mi) {
        mi = mag_lo;
    }
    digitalWriteFast(PIN_DBG_2, 0);
    digitalWriteFast(PIN_DBG_1, mag_hi - mag_lo > 0.0f);
}

FSKDemodulator::FSKDemodulator(modem_config_t *modemConfig, TeensyTimerTool::PeriodicTimer *sampleTimer,
                               unsigned int rawSize, d_raw_t *rawBuf, float *dstBuf1, float *dstBuf2,
                               adc_it_fn adc_it)
        : modem_config(modemConfig), raw_size(rawSize),
          raw_buf(rawBuf), dst_buf1(dstBuf1),
          dst_buf2(dstBuf2), sampleTimer(sampleTimer), adc_it(adc_it) {}
