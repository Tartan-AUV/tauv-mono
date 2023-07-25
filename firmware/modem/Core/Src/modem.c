//
// Created by Gleb Ryabtsev on 7/19/23.
//

#include <malloc.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

#include "modem.h"
#include "stm32f4xx_hal.h"
#include "main.h"

status_t demodulator_init(demod_t *m, const demodulator_config_t *c) {
    m->c = *c;

    int freq_sample = 250000; // todo
    m->i = 0;
    m->curr_dst_buf = m->c.dst_buf1;
    m->dst_size = m->c.raw_size / m->c.undersampling_ratio;
    m->N = m->c.raw_size + 1;
    m->dst_i = 0;

    m->c.sample_tim->Instance->ARR = SystemCoreClock / freq_sample;

    float PI = 3.14159f;
    float N = (float) m->N;
    m->k_lo = ((float) m->c.modem_config.freq_lo * (float) N) / ((float) freq_sample);
    m->k_hi = ((float) m->c.modem_config.freq_hi * (float) N) / ((float) freq_sample);

    m->coeff_w[0] = -0.25f;
    m->coeff_w[1] = 0.5f;
    m->coeff_w[2] = -0.25f;

    float r = m->c.sdft_r;
    m->coeff_a = -powf(r, (float) N);

    m->coeff_b_lo[0] = r * cexp((2.i * PI * (m->k_lo - 1.)) / (float) N);
    m->coeff_b_lo[1] = r * cexp((2.i * PI * m->k_lo) / (float) N);
    m->coeff_b_lo[2] = r * cexp((2.i * PI * (m->k_lo + 1.)) / (float) N);
    m->coeff_b_hi[0] = r * cexp((2.i * PI * (m->k_hi - 1.)) / (float) N);
    m->coeff_b_hi[1] = r * cexp((2.i * PI * m->k_hi) / (float) N);
    m->coeff_b_hi[2] = r * cexp((2.i * PI * (m->k_hi + 1.)) / (float) N);

    return MDM_OK;
}

status_t demodulator_start(demod_t *demod) {
//    __HAL_ADC_ENABLE_IT(demod->c.hadc, ADC_IT_OVR);
//    demod->c.hadc->Instance->CR2 |= ADC_CR2_DMA;
//    HAL_ADC_Start(demod->c.hadc);
//    HAL_DMAEx_MultiBufferStart_IT(demod->c.hadc->DMA_Handle, (uint32_t) &demod->c.hadc->Instance->DR,
//                                  (uint32_t) demod->raw_writing_buf, (uint32_t) demod->raw_prev_buf,
//                                  demod->c.chip_buf_size);
//    adc_start_dma_dbm(demod->c.hadc, demod->raw_writing_buf, demod->raw_last_buf, demod->c.chip_buf_size);
    HAL_ADC_Start(demod->c.hadc);
    HAL_TIM_Base_Start_IT(demod->c.sample_tim);
}


float normalize_sample(demod_t *m, d_raw_t sample) {
    return ((float) sample) / m->c.max_raw;
}

void demod_sample_it(demod_t *m) {
    demodulator_config_t *c = &m->c;
    // Wait for the end of conversion
    if (HAL_ADC_PollForConversion(&m->c.hadc, HAL_MAX_DELAY) != HAL_OK)
    {
        // ADC conversion error handling
        return; // Or any appropriate error value
    }
    uint8_t val = (uint8_t) HAL_ADC_GetValue(c->hadc);
    float sample = normalize_sample(m, val);
    float sample_N = normalize_sample(m, c->raw_buf[m->i]);

    float a = sample - m->coeff_a * sample_N;

    m->s_lo_w[0] = a + m->coeff_b_lo[0] * m->s_lo_w[0];
    m->s_lo_w[1] = a + m->coeff_b_lo[1] * m->s_lo_w[1];
    m->s_lo_w[2] = a + m->coeff_b_lo[2] * m->s_lo_w[2];

    m->s_hi_w[0] = a + m->coeff_b_hi[0] * m->s_hi_w[0];
    m->s_hi_w[1] = a + m->coeff_b_hi[1] * m->s_hi_w[1];
    m->s_hi_w[2] = a + m->coeff_b_hi[2] * m->s_hi_w[2];

    m->s_lo = m->coeff_w[0] * m->s_lo_w[0] + m->coeff_w[1] * m->s_lo_w[1] + m->coeff_w[2] * m->s_lo_w[2];
    m->s_hi = m->coeff_w[0] * m->s_hi_w[0] + m->coeff_w[1] * m->s_hi_w[1] + m->coeff_w[2] * m->s_hi_w[2];

    m->mag_lo = cabsf(m->s_lo);
    m->mag_hi = cabsf(m->s_hi);

    if(m->i % c->undersampling_ratio) {
        m->curr_dst_buf[m->dst_i] = m->mag_hi - m->mag_lo;
        m->dst_i++;
        if(m->dst_i > m->dst_size) {
            if (m->curr_dst_buf == c->dst_buf1) {
                (*c->cplt1)();
                m->curr_dst_buf = c->dst_buf2;
            }
            else {
                (*c->cplt2)();
                m->curr_dst_buf = c->dst_buf1;
            }
            m->i = 0;
            m->dst_i = 0;
        }
    }
    c->raw_buf[m->i] = val;
    m->i = (m->i + 1) % c->raw_size;

    HAL_GPIO_WritePin(DBG1_GPIO_Port, DBG1_Pin, m->mag_hi - m->mag_lo > 0.0);
}

//HAL_StatusTypeDef adc_start_dma_dbm(ADC_HandleTypeDef *hadc, uint32_t *dst0, uint32_t *dst1, uint32_t length) {
//    __IO uint32_t counter = 0U;
//    ADC_Common_TypeDef *tmpADC_Common;
//
//    /* Check the parameters */
//    assert_param(IS_FUNCTIONAL_STATE(hadc->Init.ContinuousConvMode));
//    assert_param(IS_ADC_EXT_TRIG_EDGE(hadc->Init.ExternalTrigConvEdge));
//
//    /* Process locked */
//    __HAL_LOCK(hadc);
//
//    /* Enable the ADC peripheral */
//    /* Check if ADC peripheral is disabled in order to enable it and wait during
//    Tstab time the ADC's stabilization */
//    if ((hadc->Instance->CR2 & ADC_CR2_ADON) != ADC_CR2_ADON) {
//        /* Enable the Peripheral */
//        __HAL_ADC_ENABLE(hadc);
//
//        /* Delay for ADC stabilization time */
//        /* Compute number of CPU cycles to wait for */
//        counter = (ADC_STAB_DELAY_US * (SystemCoreClock / 1000000U));
//        while (counter != 0U) {
//            counter--;
//        }
//    }
//
//    /* Check ADC DMA Mode                                                     */
//    /* - disable the DMA Mode if it is already enabled                        */
//    if ((hadc->Instance->CR2 & ADC_CR2_DMA) == ADC_CR2_DMA) {
//        CLEAR_BIT(hadc->Instance->CR2, ADC_CR2_DMA);
//    }
//
//    /* Start conversion if ADC is effectively enabled */
//    if (HAL_IS_BIT_SET(hadc->Instance->CR2, ADC_CR2_ADON)) {
//        /* Set ADC state                                                          */
//        /* - Clear state bitfield related to regular group conversion results     */
//        /* - Set state bitfield related to regular group operation                */
//                ADC_STATE_CLR_SET(hadc->State,
//                                  HAL_ADC_STATE_READY | HAL_ADC_STATE_REG_EOC | HAL_ADC_STATE_REG_OVR,
//                                  HAL_ADC_STATE_REG_BUSY);
//
//        /* If conversions on group regular are also triggering group injected,    */
//        /* update ADC state.                                                      */
//        if (READ_BIT(hadc->Instance->CR1, ADC_CR1_JAUTO) != RESET) {
//                    ADC_STATE_CLR_SET(hadc->State, HAL_ADC_STATE_INJ_EOC, HAL_ADC_STATE_INJ_BUSY);
//        }
//
//        /* State machine update: Check if an injected conversion is ongoing */
//        if (HAL_IS_BIT_SET(hadc->State, HAL_ADC_STATE_INJ_BUSY)) {
//            /* Reset ADC error code fields related to conversions on group regular */
//            CLEAR_BIT(hadc->ErrorCode, (HAL_ADC_ERROR_OVR | HAL_ADC_ERROR_DMA));
//        } else {
//            /* Reset ADC all error code fields */
//            ADC_CLEAR_ERRORCODE(hadc);
//        }
//
//        /* Process unlocked */
//        /* Unlock before starting ADC conversions: in case of potential           */
//        /* interruption, to let the process to ADC IRQ Handler.                   */
//        __HAL_UNLOCK(hadc);
//
//        /* Pointer to the common control register to which is belonging hadc    */
//        /* (Depending on STM32F4 product, there may be up to 3 ADCs and 1 common */
//        /* control register)                                                    */
//        tmpADC_Common = ADC_COMMON_REGISTER(hadc);
//
////        /* Set the DMA transfer complete callback */
////        hadc->DMA_Handle->XferCpltCallback = ADC_DMAConvCplt;
////
////        /* Set the DMA half transfer complete callback */
////        hadc->DMA_Handle->XferHalfCpltCallback = ADC_DMAHalfConvCplt;
////
////        /* Set the DMA error callback */
////        hadc->DMA_Handle->XferErrorCallback = ADC_DMAError;
////
//
//        /* Manage ADC and DMA start: ADC overrun interruption, DMA start, ADC     */
//        /* start (in case of SW start):                                           */
//
//        /* Clear regular group conversion flag and overrun flag */
//        /* (To ensure of no unknown state from potential previous ADC operations) */
//        __HAL_ADC_CLEAR_FLAG(hadc, ADC_FLAG_EOC | ADC_FLAG_OVR);
//
//        /* Enable ADC overrun interrupt */
//        __HAL_ADC_ENABLE_IT(hadc, ADC_IT_OVR);
//
//        /* Enable ADC DMA mode */
//        hadc->Instance->CR2 |= ADC_CR2_DMA;
//
//        /* Start the DMA channel */
//        HAL_DMAEx_MultiBufferStart_IT(hadc->DMA_Handle, (uint32_t) &hadc->Instance->DR, (uint32_t) dst0,
//                                      (uint32_t) dst1, length);
//
//        /* Check if Multimode enabled */
//        if (HAL_IS_BIT_CLR(tmpADC_Common->CCR, ADC_CCR_MULTI)) {
//#if defined(ADC2) && defined(ADC3)
//            if((hadc->Instance == ADC1) || ((hadc->Instance == ADC2) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_0)) \
//                                  || ((hadc->Instance == ADC3) && ((ADC->CCR & ADC_CCR_MULTI_Msk) < ADC_CCR_MULTI_4)))
//      {
//#endif /* ADC2 || ADC3 */
//            /* if no external trigger present enable software conversion of regular channels */
//            if ((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET) {
//                /* Enable the selected ADC software conversion for regular group */
//                hadc->Instance->CR2 |= (uint32_t) ADC_CR2_SWSTART;
//            }
//#if defined(ADC2) && defined(ADC3)
//            }
//#endif /* ADC2 || ADC3 */
//        } else {
//            /* if instance of handle correspond to ADC1 and  no external trigger present enable software conversion of regular channels */
//            if ((hadc->Instance == ADC1) && ((hadc->Instance->CR2 & ADC_CR2_EXTEN) == RESET)) {
//                /* Enable the selected ADC software conversion for regular group */
//                hadc->Instance->CR2 |= (uint32_t) ADC_CR2_SWSTART;
//            }
//        }
//    } else {
//        /* Update ADC state machine to error */
//        SET_BIT(hadc->State, HAL_ADC_STATE_ERROR_INTERNAL);
//
//        /* Set ADC error code to ADC IP internal error */
//        SET_BIT(hadc->ErrorCode, HAL_ADC_ERROR_INTERNAL);
//    }
//
//    /* Return function status */
//    return HAL_OK;
//}