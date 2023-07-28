//
// Created by Gleb Ryabtsev on 7/27/23.
//

#include "dsss_demodulator.h"

DSSSDemodulator::DSSSDemodulator(modem_config_t *modemConfig, const DSSSCode &code, size_t samples_per_bit) :
        modemConfig(modemConfig), code(code) {
    context_size = code.nsamples;
    context = (int8_t *) malloc(context_size);
    context_valid = false;
    memset(context, 0, context_size);

    threshold_hi = 4000;
    threshold_lo = 3000;

    lock = false;
    lock_offset = 0;
}

int32_t DSSSDemodulator::correlate(size_t pos, int8_t *buf) {
    /* Calculates a dot product of the DSSS code and the buffer with the end
     * of the code being aligned with the buffer at position pos. If pos < code.nsamples,
     * and context_valid is set then context is used to access previous samples.
     */

    int32_t sum = 0;

    int i = 0;
    for (; i < code.nsamples && i <= pos; i++) {
        sum += buf[pos - i] * code.samples[code.nsamples - i - 1];
    }

    if (!context_valid) return sum;

//    DBG_PRINT("DSSSDemodulator::correlate: context_valid = true\n");
    for (; i < code.nsamples; i++) {
        sum += context[context_size + pos - i] * code.samples[code.nsamples - i - 1];
    }

    return sum;
}

status_t DSSSDemodulator::demodulate(int8_t *buf, size_t size, uint8_t *dst, size_t dst_size, bool continuous,
                                     DemodStatus *demod_status) {

    if (!continuous) context_valid = false;

    size_t buf_i = lock ? lock_offset : 0;
    size_t dst_bit_i = 0;
    memset(dst, 0, dst_size);

    demod_status->lock_start = 0;
    demod_status->lock_end = 0;
    demod_status->intermittent_lock_loss = false;
    demod_status->read_length_bits = 0;

    if (size % code.nsamples != 0) {
        Serial.printf("DSSSDemodulator::demodulate: size %% code.nsamples != 0 (%d %% %d != 0)\n", size, code.nsamples);
        return MDM_ERROR;
    }

    DBG_PRINT("DSSSDemodulator::demodulate: START locked=%d, size = %d\n", lock, size);
    while (buf_i < size && dst_bit_i < dst_size) {
        if (lock) {
            DBG_PRINT("DSSSDemodulator::demodulate: LOCKED buf_i = %d\n", buf_i);
            int32_t c = correlate(buf_i, buf);
            DBG_PRINT("CORRELATE: %d\n", c);
            if (abs(c) < threshold_lo) {
                DBG_PRINT("DSSSDemodulator::demodulate: lock lost at %d\n", buf_i);
                lock = false;
                if (demod_status->lock_end != 0) {
                    demod_status->intermittent_lock_loss = true;
                }
                demod_status->lock_end = buf_i;
            } else {
//                DBG_PRINT("WRITING dst_bit_i = %d, c = %d\n", dst_bit_i, c);
//                dst[dst_bit_i] = c > 0;
                dst_bit_i++;
            }
            buf_i += code.nsamples;

            DBG_PRINT("IF DONE:\n");
        } else {
//            DBG_PRINT("DSSSDemodulator::demodulate: NOT LOCKED buf_i = %d, c = ", buf_i);
            int32_t c = correlate(buf_i, buf);
//            DBG_PRINT("%d\n", c);
            if (abs(c) > threshold_hi) {
                DBG_PRINT("DSSSDemodulator::demodulate: lock acquired at %d\n", buf_i);
                lock = true;
                if(!demod_status->lock_start)
                    demod_status->lock_start = buf_i;
                lock_offset = buf_i % code.nsamples;
                buf_i += code.nsamples;
            } else {
                buf_i++;
            }
        }

        if (!context_valid && buf_i > context_size) {
            if (context_size > size - buf_i) {
                Serial.printf("DSSSDemodulator::demodulate: context_size > size - buf_i (%d > %d - %d)\n",
                              context_size, size, buf_i);
                return MDM_ERROR;
            }
//            DBG_PRINT("DSSSDemodulator::demodulate: Attempting memcpy, size=%d, context_size=%d, buf_i=%d\n",
//                      size, context_size, buf_i);
            memcpy(context, &buf[size - context_size], context_size);
            DBG_PRINT("DSSSDemodulator::demodulate: memcpy done\n");
            context_valid = true;
        }
    }
    demod_status->read_length_bits = dst_bit_i + 1;
}

bool DSSSDemodulator::isLocked() {
    return lock;
}