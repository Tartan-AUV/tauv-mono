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

#define ONE_SECOND_NS 1'000'000'000

#define PIN_TX_1 18
#define REF 20
#define PIN_TX_2 23
#define PIN_RX PIN_A2
#define PIN_DBG_1 10
#define PIN_DBG_2 11

#define RAW_BUF_SIZE 200
#define SDFT_UNDERSAMPLING_RATIO 8
#define SDFT_BUF_SIZE (RAW_BUF_SIZE / SDFT_UNDERSAMPLING_RATIO)


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
