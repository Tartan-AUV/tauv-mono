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
}

Logging::~Logging() {
  // We don't delete the queue as it's static
}

bool Logging::init(UART_HandleTypeDef* uart_handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  uart_handle_ = uart_handle;
  return true;
}

void Logging::debug(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::DEBUG, format, args);
  va_end(args);
}

void Logging::info(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::INFO, format, args);
  va_end(args);
}

void Logging::warning(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::WARNING, format, args);
  va_end(args);
}

void Logging::error(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::ERROR, format, args);
  va_end(args);
}

void Logging::fatal(const char* format, ...) {
  va_list args;
  va_start(args, format);
  log(LogLevel::FATAL, format, args);
  va_end(args);
}

void Logging::log(LogLevel level, const char* format, va_list args) {
  // Skip if level is below minimum
  if (level < min_level_) {
    return;
  }
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Format header with timestamp and level
  char buffer[MAX_LOG_LENGTH];
  int header_len = snprintf(buffer, MAX_LOG_LENGTH, "[%010lu][%s] ", 
                           getTimestamp(), LEVEL_NAMES[static_cast<int>(level)]);
  
  // Format the actual message
  int msg_len = vsnprintf(buffer + header_len, MAX_LOG_LENGTH - header_len, format, args);
  
  if (msg_len < 0) {
    // Error in formatting
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
  if (result != pdPASS && uart_handle_) {
    // Try to send directly as a fallback (might block, but only as last resort)
    HAL_UART_Transmit(uart_handle_, reinterpret_cast<const uint8_t*>(message), length, 100);
  }
  
  return (result == pdPASS);
}

uint32_t Logging::getTimestamp() {
  return xTaskGetTickCount() * portTICK_PERIOD_MS;
}
