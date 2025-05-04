#pragma once

#include "stdint.h"
#include "stm32f7xx_hal.h"
#include "FreeRTOS.h"
#include "task.h"

extern uint32_t timestamp_secs;

typedef struct {
	uint32_t secs;
	uint32_t usecs;
} Timestamp_t;


static inline Timestamp_t get_timestamp() {
	taskENTER_CRITICAL();
	volatile uint32_t usecs = TIM5->CNT;
	volatile uint32_t secs = timestamp_secs;
	taskEXIT_CRITICAL();
	if (usecs >= 1000000) {
		usecs -= 1000000;
		secs += 1;
	}
	Timestamp_t res = { .secs = secs, .usecs = usecs };
	return res;
}

