//
// Created by Gleb Ryabtsev on 7/27/23.
//

#ifndef MODEM_DSSS_DEMODULATOR_H
#define MODEM_DSSS_DEMODULATOR_H

#include "main.h"
#include "dsss_modulator.h"

class DSSSDemodulator {
public:

    struct DemodStatus {
        size_t lock_start;
        size_t lock_end;
        size_t read_length_bits;
        bool intermittent_lock_loss;
    };

    DSSSDemodulator(modem_config_t *modemConfig, const DSSSCode &code, size_t samples_per_bit);

    status_t demodulate(int8_t *buf, size_t size, uint8_t *dst, size_t dst_size, bool continuous,
                        DSSSDemodulator::DemodStatus *demod_status);

    bool isLocked();



private:
    modem_config_t *modemConfig;

    int8_t *context;
    size_t context_size;
    bool context_valid;

    const DSSSCode &code;
//    size_t samples_per_bit = 7;

    uint32_t threshold_hi, threshold_lo;

//    size_t curr_buf_idx;

    int32_t correlate(size_t pos, int8_t *buf);

    bool lock;
    size_t lock_offset;
};

#endif //MODEM_DSSS_DEMODULATOR_H
