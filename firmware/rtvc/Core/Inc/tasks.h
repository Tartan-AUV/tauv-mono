/* TAUV RTVC */
/* Task Declarations & Utilities */
/* Author: Gleb Ryabtsev */

#pragma once

/* Tasks init */

void TasksInit();

/* Task function declarations */

void Task_1Hz();

void Task_10Hz();

void Task_100Hz();

void Task_1000Hz();

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
