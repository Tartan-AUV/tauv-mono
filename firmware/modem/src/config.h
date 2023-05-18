#include <Arduino.h>
#include <chrono>

#pragma once

#define ONE_S_IN_NS std::chrono::nanoseconds(1'000'000'000)
#define ONE_MS_IN_NS std::chrono::nanoseconds(1'000'000)
#define ONE_US_IN_NS std::chrono::nanoseconds(1'000)

// Parameter codes
#define FREQ_LO 0x00
#define FREQ_HI 0x01
#define FREQ_SAMPLE 0x02
#define FREQ_BIT 0x03
#define FREQ_SYNC_BIT 0x04
#define SDFT_N 0x05
#define SDFT_R 0x06
#define MAX_PAYLOAD_LENGTH 0x07
#define FILTER_COEFF 0x09
#define EDGE_THRESHOLD 0x0A
#define RECEIVE_TIMEOUT 0x0B

struct Config
{
    uint32_t freq_lo;
    uint32_t freq_hi;
    uint32_t freq_sample;
    uint32_t freq_bit;
    uint32_t freq_sync_bit;

    std::chrono::milliseconds receive_timeout;

    size_t sdft_N;
    float sdft_r;

    const std::chrono::nanoseconds half_period_lo_ns() { return ONE_S_IN_NS / (2 * freq_lo); };
    const std::chrono::nanoseconds half_period_hi_ns() { return ONE_S_IN_NS / (2 * freq_hi); };
    const std::chrono::nanoseconds period_sample_ns() { return ONE_S_IN_NS / freq_sample; };
    const std::chrono::nanoseconds period_bit_ns() { return ONE_S_IN_NS / freq_bit; };
    const std::chrono::nanoseconds period_sync_bit_ns() { return ONE_S_IN_NS / freq_sync_bit; };

    const std::chrono::nanoseconds period_sync_bit_min_ns() { return (3 * period_sync_bit_ns()) / 4; };

    uint8_t max_payload_length;

    float slow_bit_coeff;
    float fast_bit_coeff;
    float edge_threshold;

    Config(){};
};