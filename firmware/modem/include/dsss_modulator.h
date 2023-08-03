//
// Created by Gleb Ryabtsev on 7/27/23.
//

#ifndef MODEM_DSSS_MODULATOR_H
#define MODEM_DSSS_MODULATOR_H

#include "main.h"
#include "fsk_modulator.h"

struct DSSSCode {
    size_t nchips;
    uint8_t *chips;
    int8_t *samples;
    size_t nsamples;
};

struct Barker7Sequence : public DSSSCode {
    explicit Barker7Sequence(size_t samples_per_chip);
};

class DSSSModulator {
public:
    DSSSModulator(modem_config_t *modemConfig, const DSSSCode &code, FSKModulator *fskModulator,
                  size_t max_chip_buf_size);
    status_t transmit(uint8_t *buf, size_t size);
    bool busy();

private:
//    uint8_t *tx_buf;
    size_t curr_chip_buf_size;
    bool transmitting;

    DSSSCode code;
    modem_config_t modemConfig;

    FSKModulator *fskModulator;
    size_t max_chip_buf_size;
    uint8_t *chip_buf;
};
#endif //MODEM_DSSS_MODULATOR_H
