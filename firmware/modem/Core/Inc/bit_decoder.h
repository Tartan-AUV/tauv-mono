//
// Created by Alex Blass on 7/23/23.
//

#ifndef MODEM_BIT_DECODER_H
#define MODEM_BIT_DECODER_H

#include <stdbool.h>
#include "main.h"
#include "stm32f4xx_hal.h"
#include "modem.h"

#define BSEQ_LEN 50
#define BSEQ_AMPLITUDE 10
#define SLIDE_STEP 1
#define PEAKSIZE 100
#define OUTPUT_BUF_SIZE 100

typedef enum {
    PEAK_0,
    PEAK_1,
    PEAK_NOTFOUND
} decode_peak_t;

typedef struct {
    size_t alignment;
    size_t min_peaksize;
    size_t expected_peaksize;
} decoder_config_t;

typedef struct {
    decoder_config_t c;

    size_t prev_bufsize;
    uint32_t *prev_buf;
    uint32_t *output_buf;
    uint32_t b_seq[BSEQ_LEN];
} decoder_t;

/* ------------------ Library functions ------------------ */

/**
 * @brief Initialize the bit decoder's barker sequence
 * @param d Pointer to the decoder_t struct
 */
status_t bseq_init(decoder_t* d);

/**
 * @brief Initialize the bit decoder
 * @param d Pointer to the decoder_t struct
 * @return MDM_OK if successful, MDM_ERROR otherwise
 */
status_t bit_decoder_init(decoder_t* d);

/**
 * @brief Find the peak in the signal segment by calculating the correlation with the Barker sequence
 * @param dec Pointer to the decoder_t struct
 * @param seg Pointer to the signal segment
 * @return The decoded peak value (PEAK_0, PEAK_1, or PEAK_NOTFOUND)
 */
decode_peak_t find_peak(decoder_t* dec, uint32_t* seg);

/**
 * @brief Get alignment of the bit sequence
 *
 * We slide the buffer over the Barker sequence until we find two peaks; the distance between them gives
 * us the alignment of the bit sequence. We return a pointer to the beginning of the alignment, NULL if not found.
 *
 * @param dec Pointer to the decoder_t struct
 * @param raw_buf Pointer to the raw buffer
 * @param raw_buf_size Size of the raw buffer
 * @return A pointer to the beginning of the alignment, NULL if not found
 */
uint32_t* find_alignment(decoder_t* dec, uint32_t* raw_buf, size_t raw_buf_size);

/**
 * @brief Decode the bit sequence
 * @param dec Pointer to the decoder_t struct
 * @param raw_buf Pointer to the raw buffer containing the signal sequence
 * @param raw_buf_size Size of the raw buffer
 * @return MDM_OK if successful, MDM_ERROR otherwise
 */
status_t bit_decode_seq(decoder_t* dec, uint32_t* raw_buf, size_t raw_buf_size);

/**
 * @brief Deinitialize the bit decoder
 * @param dec
 * @return MDM_OK if successful, MDM_ERROR otherwise
 */
status_t bit_decoder_deinit(decoder_t* dec);

#endif //MODEM_BIT_DECODER_H
