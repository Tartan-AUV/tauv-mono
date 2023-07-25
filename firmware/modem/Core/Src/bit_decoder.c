//
// Created by Alex Blass on 7/23/23.
//

#include <memory.h>
#include <stdbool.h>
#include <math.h>
#include "bit_decoder.h"

/**
 * @brief Calculate the dot product between two sequences
 * @param a Pointer to the first sequence
 * @param b Pointer to the second sequence
 * @param size Size of the sequences
 * @return The dot product result
 */
static uint32_t dot_product(uint32_t* a, uint32_t* b, size_t size) {
    uint32_t result = 0;
    for (size_t i = 0; i < size; i++) {
        result += (a[i] * b[i]);
    }
    return result;
}

/**
 * @brief creates a waveform representing the input array
 * @param input_array
 * @param input_length
 * @param outputArray
 */
#include <stdint.h>


void create_waveform(const int8_t* input_array, size_t input_length, uint32_t* output_array) {
    if (input_array == NULL || output_array == NULL) {
        return;
    }

    uint32_t step = ((input_length - 1) << FIXED_POINT_SHIFT) / (BSEQ_LEN - 1);

    for (int i = 0; i < BSEQ_LEN; i++) {
        uint32_t pos = i * step;
        int left_ind = pos >> FIXED_POINT_SHIFT;
        int right_ind = left_ind + 1;

        int left_val = input_array[left_ind];
        int right_val = input_array[right_ind];

        uint32_t t = pos & ((1 << FIXED_POINT_SHIFT) - 1);
        int value = (left_val << FIXED_POINT_SHIFT) + t * (right_val - left_val);

        output_array[i] = (uint32_t)value * BSEQ_AMPLITUDE >> FIXED_POINT_SHIFT;
    }
}



/**
 * @brief Initialize the bit decoder's barker sequence
 * @param d
 */
void bseq_init(decoder_t *d) {
    // Barker sequence: +++--+- (length 7)
    const int8_t barker_seq[BSEQ_REF_LEN] = {1, 1, 1, -1, -1, 1, -1};

    create_waveform(barker_seq, BSEQ_REF_LEN, (uint32_t *) &d->b_seq);
}

/**
 * @brief initialize the bit decoder
 * @param d
 * @return DEC_OK if successful, DEC_ERROR otherwise
 */
dec_status_t bit_decoder_init(decoder_t *d, decoder_config_t *config) {
    if (d == NULL) {
        return DEC_ERROR;
    }

    d->c = *config;
    bseq_init(d);

    return DEC_OK;
}

/**
 * @brief Find the peak in the signal segment by calculating the correlation with the Barker sequence
 * @param dec Pointer to the decoder_t struct
 * @param seg Pointer to the signal segment
 * @return The decoded peak value (PEAK_0, PEAK_1, or PEAK_NOTFOUND)
 */
decode_peak_t find_peak(decoder_t* dec, uint32_t* seg) {
    uint32_t* seg_end = seg + dec->c.expected_peaksize;

    uint32_t max_correlation = 0;

    // Iterate through the segment and find the peak with maximum correlation
    for (uint32_t* i = seg; i < seg_end; i++) {
        uint32_t correlation = dot_product(i, dec->b_seq, BSEQ_LEN);
        if (correlation > max_correlation) {
            max_correlation = correlation;
        }
    }

    if (max_correlation >= dec->c.min_peaksize) {
        return PEAK_1;
    } else if (max_correlation <= -(dec->c.min_peaksize)) {
        return PEAK_0;
    } else {
        return PEAK_NOTFOUND;
    }
}

/**
 * @brief get alignment of the bit sequence
 *
 * We slide the buffer over the barker sequence until we find two peaks; the distance between them gives
 * us the alignment of the bit sequence. We return a pointer to the first peak.
 *
 * @param dec
 * @param raw_buf
 * @param raw_buf_size
 * @return a pointer to the beginning of the alignment, NULL if not found
 */
uint32_t *find_alignment(decoder_t *dec, uint32_t *raw_buf, size_t raw_buf_size) {
    uint32_t *seq_begin = NULL;
    size_t algn = 0;
    bool pk_found = false;

    for (uint32_t *seg = raw_buf; seg < raw_buf + raw_buf_size; seg += SLIDE_STEP) {
        decode_peak_t peak = find_peak(dec, seg);

        if (pk_found) {
            algn += SLIDE_STEP;
        }

        if (peak == PEAK_0 || peak == PEAK_1) {
            seq_begin = seg;

            if (pk_found) {
                dec->c.alignment = algn;
                break;
            }
            pk_found = true;
        }
    }

    return seq_begin;
}

/**
 * @brief decode the bit sequence
 * @param dec
 * @param raw_buf
 * @param raw_buf_size
 * @return DEC_OK if successful, DEC_ERROR otherwise
 */
dec_status_t bit_decode_seq(decoder_t *dec, uint32_t *raw_buf, size_t raw_buf_size) {
    if (dec == NULL || raw_buf == NULL) {
        return DEC_ERROR;
    }

    uint32_t seq[raw_buf_size + dec->prev_bufsize];
    uint32_t *seq_begin = (uint32_t *) &seq;

    if (dec->using_prev_buf) {
        // If prev_buf non-NULL, we haven't finished reading last buffer
        // In this case, we alloc new buffer and memcpy contents of both buffers over

        memcpy(&seq, dec->prev_buf, (dec->prev_bufsize * sizeof(uint32_t)));
        memcpy(&seq + dec->prev_bufsize, raw_buf, (raw_buf_size * sizeof(uint32_t)));

        raw_buf_size = dec->prev_bufsize + raw_buf_size;
    } else if (dec->c.alignment == 0) {
        seq_begin = find_alignment(dec, seq_begin, raw_buf_size);
    } else {
        seq_begin = raw_buf;
    }

    //Enter the main decode loop
    for (uint32_t *seg = seq_begin; seg < seq_begin + raw_buf_size; seg += dec->c.alignment) {
        decode_peak_t peak = find_peak(dec, seg);

        if (peak == PEAK_0) {
            //TODO: Handle 0
        } else if (peak == PEAK_1) {
            //@TODO: Handle 1
        } else {
            // No peak found, realign
            seg = find_alignment(dec, seg, raw_buf_size - (seg - seq_begin));
            if (seg == NULL) {
                // We couldn't realign, so we need to save the rest of the buffer for next time
                dec->c.alignment = 0;
                memcpy(&dec->prev_buf, seg, (raw_buf_size - (seg - seq_begin)) * sizeof(uint32_t));
                break;
            }
        }

        // If we don't have enough buffer for alignment, save the rest for next time
        if (raw_buf_size - (seg - seq_begin) < dec->c.alignment) {
            memcpy(dec->prev_buf, seg, (raw_buf_size - (seg - seq_begin)) * sizeof(uint32_t));
            dec->prev_bufsize = raw_buf_size - (seg - seq_begin);
            break;
        }
    }


    return DEC_OK;
}

dec_status_t bit_decoder_deinit(decoder_t *d) {
    if (d == NULL) {
        return DEC_OK;
    }

    return DEC_OK;
}

