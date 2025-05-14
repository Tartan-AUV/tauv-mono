/*
 * logging.c
 *
 *  Created on: Apr 12, 2025
 *      Author: gleb
 */

#include "logging.h"

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>

#include "stm32f7xx_hal.h"
#include "cmsis_os.h"
#include "netif.h"

// Private typedefs
typedef struct {
	char msg[LOG_MSG_MAX_LEN];
	size_t len;
	LogLevel_t level;
} LogMessage_t;

// Static variables
static QueueHandle_t 		logQueueSerial;
static QueueHandle_t 		logQueueEth;
static StaticQueue_t 		logQueueSerialStruct;
static StaticQueue_t 		logQueueEthStruct;
static uint8_t 				logQueueSerialBuf[LOG_QUEUE_LEN * sizeof(LogMessage_t)];
static uint8_t 				logQueueEthBuf[LOG_QUEUE_LEN * sizeof(LogMessage_t)];
static UART_HandleTypeDef  	*logUartHandle_p = NULL;
static osThreadId			logSerialTaskHandle;
static osThreadId			logEthTaskHandle;


SemaphoreHandle_t serialMutex;

// Function prototypes
void LogSerialTask();
void LogEthernetTask();


void LogInitSerial(UART_HandleTypeDef *huart)
{
	assert (huart);

	logUartHandle_p = huart;
    serialMutex = xSemaphoreCreateMutex();
    logQueueSerial = xQueueCreateStatic(LOG_QUEUE_LEN,
    				   	   	   	        sizeof(LogMessage_t),
										logQueueSerialBuf,
									    &logQueueSerialStruct);

	osThreadDef(logSerialTask, LogSerialTask, osPriorityBelowNormal, 0, 512);
	logSerialTaskHandle = osThreadCreate(osThread(logSerialTask), NULL);

    LogPrintf(LOG_DEST_SERIAL, LOG_LEVEL_DEBUG, "Serial logging initialized.");
}

void LogInitEthernet()
{
    logQueueEth = xQueueCreateStatic(LOG_QUEUE_LEN,
    				   	   	   	     sizeof(LogMessage_t),
									 logQueueEthBuf,
									 &logQueueEthStruct);

//    if (!netif_is_up(&gnetif) || !netif_is_link_up(&gnetif)) {
//		LogPrintf(LOG_DEST_SERIAL, LOG_LEVEL_ERROR, "Could not initialize Ethernet logging, interface down.");
//    }
	osThreadDef(logEthTask, LogEthernetTask, osPriorityBelowNormal, 0, 512);
	logEthTaskHandle = osThreadCreate(osThread(logEthTask), NULL);

	LogPrintf(LOG_DEST_SERIAL, LOG_LEVEL_INFO, "Ethernet logging initialized.");
}

void LogPrintf(LogDest_t dest, LogLevel_t level, char *fmt, ...)
{
    va_list args;
    char buffer[LOG_MSG_MAX_LEN];

    va_start(args, fmt);
    int len = vsnprintf(buffer, LOG_MSG_MAX_LEN, fmt, args);
    va_end(args);

    LogMessage_t msg;

    if (len < 0) {
        // Formatting error, just send an empty message
        msg.msg[0] = '\0';
        msg.len = 0;
    } else if (len >= LOG_MSG_MAX_LEN) {
        // Message was truncated
        strncpy(msg.msg, buffer, LOG_MSG_MAX_LEN - 1);
        msg.msg[LOG_MSG_MAX_LEN - 1] = '\0';
        msg.len = LOG_MSG_MAX_LEN - 1;
    } else {
        strcpy(msg.msg, buffer);
        msg.len = len;
    }

    msg.level = level;

    if (dest & LOG_DEST_SERIAL)
    {
		xQueueSend(logQueueSerial, &msg, 0);
    }

    if (dest & LOG_DEST_ETH)
    {
    	xQueueSend(logQueueEth, &msg, 0);
    }

}

void LogSerialTask()
{
	assert (logUartHandle_p);

    LogMessage_t msg;

    while (1) {
        if (xQueueReceive(logQueueSerial, &msg, portMAX_DELAY) == pdTRUE) {
            const char *levelStr = "[UNK] ";

            switch (msg.level) {
                case LOG_LEVEL_ERROR: levelStr = "[ERR] "; break;
                case LOG_LEVEL_WARN:  levelStr = "[WARN] "; break;
                case LOG_LEVEL_INFO:  levelStr = "[INFO] "; break;
                case LOG_LEVEL_DEBUG: levelStr = "[DBG] "; break;
            }

            HAL_UART_Transmit(logUartHandle_p, (uint8_t *)levelStr, strlen(levelStr), HAL_MAX_DELAY);
            HAL_UART_Transmit(logUartHandle_p, (uint8_t *)msg.msg, msg.len, HAL_MAX_DELAY);
        }
    }
}

void LogEthernetTask()
{
	// todo: implement me
	assert(0);
	while (1)
	{

	}
}

int _write(int file, char *data, int len)  // override default syscall
{
	assert(logUartHandle_p);
    if (serialMutex != NULL) {
        if (xSemaphoreTake(serialMutex, portMAX_DELAY) == pdTRUE) {
            HAL_UART_Transmit(logUartHandle_p, (uint8_t*)data, len, HAL_MAX_DELAY);
            xSemaphoreGive(serialMutex);
        }
    } else {
        HAL_UART_Transmit(logUartHandle_p, (uint8_t*)data, len, HAL_MAX_DELAY);
    }

    return len;
}
