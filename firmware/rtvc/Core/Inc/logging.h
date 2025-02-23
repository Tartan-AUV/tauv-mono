/*
 * logging.h
 *
 *  Created on: Apr 12, 2025
 *      Author: gleb
 */

#ifndef INC_LOGGING_H_
#define INC_LOGGING_H_

#include "stm32f7xx_hal.h"

// General configuration
#define LOG_MSG_MAX_LEN 128
#define LOG_QUEUE_LEN	16

// Destination
typedef uint8_t LogDest_t;
#define LOG_DEST_ETH 	0x01
#define LOG_DEST_SERIAL 0x02
#define LOG_DEST_ALL 	0x04

// Logging levels
typedef uint8_t LogLevel_t;
#define LOG_LEVEL_NONE   0
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_DEBUG  4

// Set current logging level here
#define LOG_LEVEL 			LOG_LEVEL_DEBUG
#define LOG_DEST_DEFAULT 	LOG_DEST_SERIAL
#define LOG_DEST_IP

// Logging macros
#if LOG_LEVEL >= LOG_LEVEL_ERROR
#define ERROR(fmt, ...)   LogPrintf(LOG_DEST_DEFAULT, LOG_LEVEL_ERROR, fmt "\r\n", ##__VA_ARGS__)
#else
#define ERROR(fmt, ...)   ((void)0)
#endif

#if LOG_LEVEL >= LOG_LEVEL_WARN
#define WARN(fmt, ...)    LogPrintf(LOG_DEST_DEFAULT, LOG_LEVEL_WARN, fmt "\r\n", ##__VA_ARGS__)
#else
#define WARN(fmt, ...)    ((void)0)
#endif

#if LOG_LEVEL >= LOG_LEVEL_INFO
#define INFO(fmt, ...)    LogPrintf(LOG_DEST_DEFAULT, LOG_LEVEL_INFO, fmt "\r\n", ##__VA_ARGS__)
#else
#define INFO(fmt, ...)    ((void)0)
#endif

#if LOG_LEVEL >= LOG_LEVEL_DEBUG
#define DBG(fmt, ...)     LogPrintf(LOG_DEST_DEFAULT, LOG_LEVEL_DEBUG, fmt "\r\n", ##__VA_ARGS__)
#else
#define DBG(fmt, ...)     ((void)0)
#endif

void LogInitSerial(UART_HandleTypeDef *huart);

void LogInitEthernet();

void LogPrintf(LogDest_t dest, LogLevel_t level, char *fmt, ...);


#endif /* INC_LOGGING_H_ */
