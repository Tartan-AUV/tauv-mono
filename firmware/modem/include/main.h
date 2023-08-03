//
// Created by Gleb Ryabtsev on 7/25/23.
//

#ifndef MODEM_MAIN_H
#define MODEM_MAIN_H

#include <Arduino.h>
#undef abs

#include <ADC.h>
#include <TeensyTimerTool.h>
#include <complex.h>
#include <math.h>

#define DBG

#ifdef DBG
#define DBG_PRINT(...) Serial.printf(__VA_ARGS__)
#else
#define DBG_PRINT(...)
#endif

#define ONE_SECOND_NS 1'000'000'000

#define PIN_TX_1 18
#define REF 20
#define PIN_TX_2 23
#define PIN_RX PIN_A2
#define PIN_DBG_1 36
#define PIN_DBG_2 37

#define PACKET_SIZE 8
#define HEADER_SIZE 2
#define RX_BUF_SIZE (PACKET_SIZE+HEADER_SIZE) * 2
#define TX_BUF_SIZE (PACKET_SIZE+HEADER_SIZE)

#define CHIP_RATE 500
#define SEQUENCE_LENGTH 7
#define ADC_SAMPLING_FREQ 200000

#define SAMPLES_PER_CHIP 8
#define RAW_BUF_SIZE (ADC_SAMPLING_FREQ / CHIP_RATE) * (2 / 4)
#define SDFT_UNDERSAMPLING_RATIO ((ADC_SAMPLING_FREQ / CHIP_RATE) / SAMPLES_PER_CHIP)

#define SDFT_OUT_BUF_SIZE (RX_BUF_SIZE * 8 * SEQUENCE_LENGTH * SAMPLES_PER_CHIP)

typedef enum {
    MDM_OK,
    MDM_ERROR
} status_t;
//
typedef struct {
    int freq_lo;
    int freq_hi;
    int chip_rate;
} modem_config_t;
#endif //MODEM_MAIN_H
