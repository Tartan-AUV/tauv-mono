//
// Created by Gleb Ryabtsev on 7/22/23.
//

#include <chrono>

#include "fsk_modulator.h"
#include "main.h"

//#include "math.h"

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

status_t FSKModulator::init() {
    pinMode(PIN_TX_1, OUTPUT);
    pinMode(PIN_TX_2, OUTPUT);
    digitalWriteFast(PIN_TX_1, 0);
    digitalWriteFast(PIN_TX_1, 1);
    /* Generate LUTs */
    // normalize and map to the half-periods
    float k = 1.0f / gaussian_integral(1.0f, sigma);
    uint32_t hp_lo =  ONE_SECOND_NS / (modem_config->freq_lo * 2);
    uint32_t hp_hi = ONE_SECOND_NS / (modem_config->freq_hi * 2);
    float m = ((float)hp_hi - (float)hp_lo) / 2.0f;
    float n = ((float) (hp_lo + hp_hi)) / 2.0f;
    for (int i = 0; i < FREQ_LUT_SIZE; i++) {
        float x = -1.0f + 2.0f * ((float) i) / FREQ_LUT_SIZE;
        float g = k * gaussian_integral(x, sigma);
        period_lut[i] = (uint32_t) (m*g + n);
    }

    bit_period = ONE_SECOND_NS / modem_config->chip_rate;

    lut_time_coeff = (float) FREQ_LUT_SIZE / (float) bit_period;

    transmitting = false;
    buf = NULL;
    buf_size = 0;
//    buf_time_end = 0;
//    buf_time = 0;

    period_hi = (ONE_SECOND_NS / modem_config->freq_hi) / 2;
    period_lo = (ONE_SECOND_NS / modem_config->freq_lo) / 2;

//    Serial.printf("%ld %ld\n", period_hi, period_lo);

    return MDM_OK;
}

status_t FSKModulator::transmit(m_word_t *buf, size_t size) {
    this->buf = buf;
    buf_size = size;

    transmitting = true;
//    prev_bit = tx_buf[0] & 0x1;
//    curr_bit = tx_buf[0] & 0x1;
//    buf_time = 0;
//    buf_time_end = bit_period * (size * 8);
//    transmitting = true;
//    next_bit_index = 1;
    curr_bit_i = 0;
//    auto cb_gaussian = [this] { this->FSKModulator::fsk_mod_tim_it(); };
    auto carrier_cb = [this] {this->FSKModulator::carrier_timer_isr(); };
    auto bit_cb = [this] {this->FSKModulator::bit_timer_isr(); };

    digitalWriteFast(PIN_TX_1, 1);
    digitalWriteFast(PIN_TX_2, 0);
    carrier_timer->begin(carrier_cb, std::chrono::nanoseconds(period_lo), true);
    bit_timer->begin(bit_cb, std::chrono::nanoseconds(bit_period), true);

    return MDM_OK;
}

bool FSKModulator::busy() {
    return transmitting;
}

void FSKModulator::carrier_timer_isr() {
    digitalWriteFast(PIN_TX_1, !digitalReadFast(PIN_TX_1));
    digitalWriteFast(PIN_TX_2, !digitalReadFast(PIN_TX_1));
}

void FSKModulator::bit_timer_isr() {
    size_t curr_byte_i = curr_bit_i / 8;
    uint8_t msk = 0x1 << (curr_bit_i % 8);
//    Serial.println("bit isr");
    if (curr_byte_i >= buf_size) {
        carrier_timer->stop();
        bit_timer->stop();
        digitalWriteFast(PIN_TX_1, 0);
        digitalWriteFast(PIN_TX_2, 0);
        transmitting = false;
        return;
    }
    bool curr_bit = buf[curr_byte_i] & msk;
    carrier_timer->setPeriod(std::chrono::nanoseconds (curr_bit ? period_hi : period_lo));
    curr_bit_i++;
}

//void FSKModulator::fsk_mod_tim_it() {
//    if (buf_time >= buf_time_end) {
//        carrier_timer->stop();
//        digitalWriteFast(PIN_TX_1, 0);
//        digitalWriteFast(PIN_TX_2, 0);
//        transmitting = false;
//        return;
//    }
//
//
//    digitalWriteFast(PIN_TX_1, !digitalReadFast(PIN_TX_1));
//    digitalWriteFast(PIN_TX_2, !digitalReadFast(PIN_TX_1));
//
//
//    size_t next_bit_i = buf_time / bit_period + 1; // can go 1 bit past the buffer
//
//    if (next_bit_i != next_bit_index) {
//
//        next_bit_index = next_bit_i;
//        prev_bit = curr_bit;
//        curr_bit = next_bit;
//
//        next_bit_i = min(next_bit_i, buf_size * 8 - 1);
//
//        size_t next_byte_i = next_bit_i / 8;
//        uint8_t msk = 0x1 << (next_bit_i % 8);
//        next_bit = tx_buf[next_byte_i] & msk;
//
//    }
//
//    uint32_t T = (uint32_t) ((float) ( buf_time % bit_period) * lut_time_coeff); // TODO: replace with int div
//
//    uint32_t period;
//    if (T < FREQ_LUT_SIZE / 2) {
//        if ( prev_bit && ! curr_bit) {
//            period = period_lut[FREQ_LUT_SIZE / 2 - T - 1];
//        } else if (! prev_bit && curr_bit){
//            period = period_lut[FREQ_LUT_SIZE / 2 + T];
//        } else {
//            size_t idx = curr_bit ? FREQ_LUT_SIZE - 1 : 0;
//            period = period_lut[idx];
//        }
//    } else {
//        if (! curr_bit && next_bit) {
//            period = period_lut[T - FREQ_LUT_SIZE / 2];
//        } else if ( curr_bit & ! next_bit) {
//            period = period_lut[FREQ_LUT_SIZE * 3 / 2 - 1 - T];
//        } else {
//            size_t idx = curr_bit ? FREQ_LUT_SIZE - 1 : 0;
//            period = period_lut[idx];
//        }
//    }
//    carrier_timer->setPeriod(std::chrono::nanoseconds(period));
//    buf_time += period;
//}

FSKModulator::FSKModulator(modem_config_t *modemConfig, TeensyTimerTool::PeriodicTimer *t,
                           TeensyTimerTool::PeriodicTimer *t2) : modem_config(modemConfig),
                                                                 carrier_timer(t),
                                                                 bit_timer(t2){}

void FSKModulator::setSigma(float sigma) {
    FSKModulator::sigma = sigma;
}
