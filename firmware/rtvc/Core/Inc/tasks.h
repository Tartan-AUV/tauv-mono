/* TAUV RTVC */
/* Task Declarations & Utilities */
/* Author: Gleb Ryabtsev */

#pragma once

/* Task function declarations */

void task1hz();

void task10hz();

void task100hz();

void task1000hz();

/* Same-frequency buffer structures */

typedef struct {} MessageBuffer1Hz;
typedef struct {} MessageBuffer10Hz;
typedef struct {} MessageBuffer100Hz;
typedef struct {} MessageBuffer1000Hz;

/* Inter-frequency buffer structures */

/* Utilities */

typedef enum TaskStatus {
    TASK_RESULT_OK,
    TASK_RESULT_ERR
} RegularTaskStatus_t;
