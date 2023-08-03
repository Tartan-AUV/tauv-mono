//
// Created by Gleb Ryabtsev on 7/27/23.
//

#include "dsss_modulator.h"

Barker7Sequence::Barker7Sequence(size_t samples_per_chip) : DSSSCode() {
    nchips = 7;
    chips = new uint8_t [nchips] {1, 1, 1, 0, 0, 1, 0};
    samples = new int8_t [nchips * samples_per_chip];

    for (int i = 0; i < nchips; i++) {
        for (int j = 0; j < samples_per_chip; j++) {
            samples[i*samples_per_chip + j] = chips[i] ? 1 : -1;
        }
    }

    nsamples = nchips * samples_per_chip;
}

DSSSModulator::DSSSModulator(modem_config_t *modemConfig, const DSSSCode &code, FSKModulator *fskModulator,
                             size_t max_chip_buf_size) :
        modemConfig(*modemConfig), max_chip_buf_size(max_chip_buf_size), code(code), fskModulator(fskModulator) {
    transmitting = false;
    curr_chip_buf_size = 0;
    chip_buf = new uint8_t[max_chip_buf_size];
}

status_t DSSSModulator::transmit(uint8_t *buf, size_t size) {
    // create a new chip buffer using the code, and transmit using fsk_modulator

    if (size * code.nchips + 1> max_chip_buf_size) {
        Serial.println("DSSSModulator::transmit: chip buffer too small");
        return MDM_ERROR;
    }

    memset(chip_buf, 0, size * code.nchips + 1);

    // Serial.printf("DSSSModulator::transmit: size = %d, nchips = %d\n", size, code.nchips);
    size_t chip_buf_chip_i = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 8; j++) {
            bool bit = (buf[i] >> j) & 1;
            for (int k = 0; k < code.nchips; k++) {
                size_t chip_buf_byte = chip_buf_chip_i / 8;
                uint8_t chip = bit ? code.chips[k] : !code.chips[k];
                size_t shifted_chip = chip << (chip_buf_chip_i % 8);
                chip_buf[chip_buf_byte] |= shifted_chip;
                chip_buf_chip_i++;
            }
        }
    }

    while (fskModulator->busy())
        ;
    // Serial.println("DSSSModulator::transmit: transmitting");

    return fskModulator->transmit(chip_buf, size * code.nchips);
}

bool DSSSModulator::busy() {
    return fskModulator->busy();
}