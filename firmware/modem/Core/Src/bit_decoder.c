//
// Created by Alex Blass on 7/23/23.
//

#include <malloc.h>
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
 * @brief A function that returns an array of length len that is a quadratic approximation of the input array
 *
 * The input array should consist of only 1s and -1s. We create a new array of length BSEQ_LEN that is a quadratic
 * approximation of the input array. The quadratic approximation is a parabola that goes from 0 to BSEQ_AMPLITUDE and
 * then back to 0. The quadratic approximation is used to create the barker sequence.
 */
static uint32_t *quad_approximation(int *seq, size_t seq_len, size_t output_len, uint32_t amplitude) {
    uint32_t *output = malloc(output_len * sizeof(uint32_t));
    if (output == NULL) {
        return NULL;
    }

    // Find the indexes of 1s and -1s in the input sequence
    size_t num_ones = 0, num_neg_ones = 0;
    for (size_t i = 0; i < seq_len; i++) {
        seq[i] == 1 ? num_ones++ : num_neg_ones++;
    }

    // Calculate the step size for each part of the quadratic approximation
    double step_size = (double)(output_len - 1) / (num_ones + num_neg_ones);

    // Create the quadratic approximation based on the number of 1s and -1s
    size_t index = 0;
    for (size_t i = 0; i < seq_len; i++) {
        if (seq[i] == 1) {
            for (size_t j = 0; j < step_size; j++) {
                double x = (double)index / (output_len - 1);
                double y = -4 * pow(x - 0.5, 2) + 1;
                output[index++] = (uint32_t)(y * amplitude);
            }
        } else if (seq[i] == -1) {
            for (size_t j = 0; j < step_size; j++) {
                double x = (double)index / (output_len - 1);
                double y = 4 * pow(x - 0.5, 2) - 1;
                output[index++] = (uint32_t)(y * amplitude);
            }
        }
    }

    return output;
}


/**
 * @brief Initialize the bit decoder's barker sequence
 * @param d
 */
status_t bseq_init(decoder_t *d) {
    uint32_t *b_seq = d->b_seq;

    // Barker sequence: +++--+- (length 7)
    int barker_seq[7] = {1, 1, 1, -1, -1, 1, -1};

    b_seq = quad_approximation(barker_seq, 7, BSEQ_LEN, BSEQ_AMPLITUDE);

    return b_seq == NULL ? MDM_ERROR : MDM_OK;
}

/**
 * @brief initialize the bit decoder
 * @param d
 * @return MDM_OK if successful, MDM_ERROR otherwise
 */
status_t bit_decoder_init(decoder_t *d) {
    if (d == NULL) {
        return MDM_ERROR;
    }

    decoder_config_t config;

    config.alignment = 0;
    config.expected_peaksize = PEAKSIZE;
    config.min_peaksize = PEAKSIZE/1.2;

    d->c = config;
    d->prev_buf = NULL;

    return bseq_init(d);
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
    int32_t peaksize = 0;

    // Iterate through the segment and find the peak with maximum correlation
    for (uint32_t* i = seg; i < seg_end; i++) {
        uint32_t correlation = dot_product(i, dec->b_seq, BSEQ_LEN);
        if (correlation > max_correlation) {
            max_correlation = correlation;
            peaksize = i - seg;
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
 * @return MDM_OK if successful, MDM_ERROR otherwise
 */
status_t bit_decode_seq(decoder_t *dec, uint32_t *raw_buf, size_t raw_buf_size) {
    if (dec == NULL || raw_buf == NULL) {
        return MDM_ERROR;
    }

    uint32_t *seq_begin;

    if (dec->prev_buf != NULL) {
        // If prev_buf non-NULL, we haven't finished reading last buffer
        // In this case, we alloc new buffer and memcpy contents of both buffers over
        seq_begin = (uint32_t *) malloc((dec->prev_bufsize + raw_buf_size) * sizeof(uint32_t));

        memcpy(seq_begin, dec->prev_buf, (dec->prev_bufsize * sizeof(uint32_t)));
        memcpy(seq_begin + dec->prev_bufsize, raw_buf, (raw_buf_size * sizeof(uint32_t)));

        //@TODO: Do we need to free raw_buf here?
        free(dec->prev_buf);
        dec->prev_buf = NULL;

        raw_buf_size = dec->prev_bufsize + raw_buf_size;
    } else if (dec->c.alignment == 0) {
        seq_begin = find_alignment(dec, raw_buf, raw_buf_size);
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
                dec->prev_buf = (uint32_t *) malloc((raw_buf_size - (seg - seq_begin)) * sizeof(uint32_t));
                break;
            }
        }

        // If we don't have enough buffer for alignment, save the rest for next time
        if (raw_buf_size - (seg - seq_begin) < dec->c.alignment) {
            dec->prev_buf = (uint32_t *) malloc((raw_buf_size - (seg - seq_begin)) * sizeof(uint32_t));
            memcpy(dec->prev_buf, seg, (raw_buf_size - (seg - seq_begin)) * sizeof(uint32_t));
            dec->prev_bufsize = raw_buf_size - (seg - seq_begin);
            break;
        }
    }


    return MDM_OK;
}

status_t bit_decoder_deinit(decoder_t *d) {
    if (d == NULL) {
        return MDM_OK;
    }

    if (d->prev_buf != NULL) {
        free(d->prev_buf);
    }

    return MDM_OK;
}

