/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      Implementation of the logging utility.
 *
 *****************************************************************************/

#include "FreeRTOS.h"
#include "task.h"
#include <cstdio>
#include <cstring>

#include "Logging.hpp"

// ITM_SendChar declaration (defined in LoggingTask.cpp)
extern "C" uint32_t ITM_SendChar(uint32_t ch);

using namespace TAUV;

// Initialize static queue members
StaticQueue_t Logging::queue_buffer_;
uint8_t Logging::queue_storage_area_[LOG_QUEUE_SIZE * sizeof(LogMessage)];
QueueHandle_t Logging::message_queue_ = nullptr;

Logging::Logging() {
  // Create the message queue if it hasn't been created yet
  if (message_queue_ == nullptr) {
    message_queue_ = xQueueCreateStatic(
      LOG_QUEUE_SIZE,
      sizeof(LogMessage),
      queue_storage_area_,
      &queue_buffer_
    );
  }
  
  // Create the mutex using static allocation
  mutex_ = xSemaphoreCreateMutexStatic(&mutex_buffer_);
}

Logging::~Logging() {
  // We don't delete the queue as it's static
}

void Logging::setLogLevel(LogLevel level) {
  if (xSemaphoreTake(mutex_, portMAX_DELAY) == pdTRUE) {
    min_level_ = level;
    xSemaphoreGive(mutex_);
  }
}

LogLevel Logging::getLogLevel() const {
  return min_level_;
}

LogOutputMode Logging::getOutputMode() const {
  return output_mode_;
}

UART_HandleTypeDef* Logging::getUartHandle() const {
  return uart_handle_;
}

bool Logging::init(UART_HandleTypeDef* uart_handle, LogOutputMode mode) {
  if (xSemaphoreTake(mutex_, portMAX_DELAY) == pdTRUE) {
    uart_handle_ = uart_handle;
    output_mode_ = mode;
    xSemaphoreGive(mutex_);
    return true;
  }
  return false;
}

void Logging::debug(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::LOG_LEVEL_DEBUG, format, args);
  va_end(args);
}

void Logging::info(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::LOG_LEVEL_INFO, format, args);
  va_end(args);
}

void Logging::warning(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::LOG_LEVEL_WARNING, format, args);
  va_end(args);
}

void Logging::error(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::LOG_LEVEL_ERROR, format, args);
  va_end(args);
}

void Logging::fatal(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::LOG_LEVEL_FATAL, format, args);
  va_end(args);
}

void Logging::log(LogLevel level, const char* format, va_list args) {
  // Skip if level is below minimum
  if (level < min_level_) {
    return;
  }
  
  if (xSemaphoreTake(mutex_, portMAX_DELAY) == pdTRUE) {
  
  // Format header with timestamp and level
  char buffer[MAX_LOG_LENGTH];
  int header_len = snprintf(buffer, MAX_LOG_LENGTH, "[%010lu][%s] ", 
                           getTimestamp(), LEVEL_NAMES[static_cast<int>(level)]);
  
  // Format the actual message
  int msg_len = vsnprintf(buffer + header_len, MAX_LOG_LENGTH - header_len, format, args);
  
  if (msg_len < 0) {
    // Error in formatting
    xSemaphoreGive(mutex_);
    return;
  }
  
  // Add newline if needed
  int total_len = header_len + msg_len;
  if (total_len + 2 < MAX_LOG_LENGTH && buffer[total_len - 1] != '\n') {
    buffer[total_len] = '\r';
    buffer[total_len + 1] = '\n';
    buffer[total_len + 2] = '\0';
    total_len += 2;
  }
  
  // Queue the message
  queueMessage(buffer, total_len);
  
  xSemaphoreGive(mutex_);
  }
}

bool Logging::queueMessage(const char* message, size_t length) {
  if (!message_queue_) {
    return false;
  }
  
  LogMessage log_msg;
  memset(log_msg.buffer, 0, sizeof(log_msg.buffer));
  
  // Ensure we don't exceed buffer size
  if (length >= sizeof(log_msg.buffer)) {
    length = sizeof(log_msg.buffer) - 1;
  }
  
  // Copy message to log_msg buffer
  memcpy(log_msg.buffer, message, length);
  log_msg.length = length;
  
  // Send to queue with timeout (non-blocking)
  BaseType_t result = xQueueSend(message_queue_, &log_msg, QUEUE_SEND_TIMEOUT);
  
  // If queue is full and we couldn't send, make a last-ditch effort to log directly
  if (result != pdPASS) {
    // Fallback logging based on configured output mode
    if ((output_mode_ == LogOutputMode::LOG_OUTPUT_MODE_UART || output_mode_ == LogOutputMode::LOG_OUTPUT_MODE_ITM_UART) &&
        uart_handle_ != nullptr) {
      // Try to send directly via UART as a fallback (might block, but only as last resort)
      HAL_UART_Transmit(uart_handle_, reinterpret_cast<const uint8_t*>(message), length, 100);
    }
    
    if (output_mode_ == LogOutputMode::LOG_OUTPUT_MODE_ITM || output_mode_ == LogOutputMode::LOG_OUTPUT_MODE_ITM_UART) {
      // Send directly via ITM as a fallback
      for (size_t i = 0; i < length; i++) {
        ITM_SendChar(message[i]);
      }
    }
  }
  
  return (result == pdPASS);
}

uint32_t Logging::getTimestamp() {
  return xTaskGetTickCount() * portTICK_PERIOD_MS;
}
