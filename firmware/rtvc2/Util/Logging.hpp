/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      Thread-safe logging utility that provides formatted log messages
 *      with different severity levels. Messages are queued and processed
 *      by a dedicated logging task to avoid blocking application threads.
 *
 *****************************************************************************/

#pragma once

#include <cstdarg>
#include <string>
#include <array>
#include <memory>

extern "C" {
#include "stm32f7xx_hal.h"
#include "FreeRTOS.h"
#include "queue.h"
#include "semphr.h"
#include "cmsis_os.h"
}

namespace TAUV {

enum class LogLevel {
  LOG_LEVEL_DEBUG,
  LOG_LEVEL_INFO,
  LOG_LEVEL_WARNING,
  LOG_LEVEL_ERROR,
  LOG_LEVEL_FATAL,
  LOG_LEVEL_NONE  // Special value to disable logging
};

// Defines output mode for logging
enum class LogOutputMode {
  LOG_OUTPUT_MODE_UART,   // Output to UART
  LOG_OUTPUT_MODE_ITM,    // Output to ITM (SWD/SWDIO)
  LOG_OUTPUT_MODE_ITM_UART    // Output to both UART and ITM
};

// Structure to hold a log message
struct LogMessage {
  char buffer[256];
  size_t length;
};

class Logging {
public:
  static constexpr size_t MAX_LOG_LENGTH = 256;
  static constexpr size_t LOG_QUEUE_SIZE = 16;  // Number of messages in queue
  static constexpr TickType_t QUEUE_SEND_TIMEOUT = 0;  // Don't block if queue is full (0 = no wait)

  // Get access to the log message queue
  static QueueHandle_t getMessageQueue() { return message_queue_; }

  // Singleton accessor
  static Logging& getInstance() {
    static Logging instance;
    return instance;
  }

  // Initialize the logging system with specified UART and output mode
  bool init(UART_HandleTypeDef* uart_handle, LogOutputMode mode = LogOutputMode::LOG_OUTPUT_MODE_UART);

  // Set the minimum log level
  void setLogLevel(LogLevel level);

  // Get the current log level
  LogLevel getLogLevel() const;

  // Get current output mode
  LogOutputMode getOutputMode() const;

  // Log functions for different severity levels
  void debug(const char* format, ...);
  void info(const char* format, ...);
  void warning(const char* format, ...);
  void error(const char* format, ...);
  void fatal(const char* format, ...);

  // Generic printf-style function with configurable log level
  void printf(LogLevel level, const char* format, ...);
  
  // Raw printf-style function without severity marker
  void rawPrintf(const char* format, ...);
  
  // Helper method for raw logging without severity (needed for printf)
  void logRaw(const char* format, va_list args);
  
  // Get UART handle
  UART_HandleTypeDef* getUartHandle() const;
  
  private:
  Logging();
  ~Logging();
  
  // Prevent copying and assignment
  Logging(const Logging&) = delete;
  Logging& operator=(const Logging&) = delete;
  
  // Helper method for formatted logging
  void log(LogLevel level, const char* format, va_list args);

  // Queue a log message to be processed by the logging task
  bool queueMessage(const char* message, size_t length);

  // Static queue for log messages
  static StaticQueue_t queue_buffer_;
  static uint8_t queue_storage_area_[LOG_QUEUE_SIZE * sizeof(LogMessage)];
  static QueueHandle_t message_queue_;

  // FreeRTOS mutex for thread safety
  SemaphoreHandle_t mutex_;
  StaticSemaphore_t mutex_buffer_;
  LogLevel min_level_ = LogLevel::LOG_LEVEL_INFO;
  LogOutputMode output_mode_ = LogOutputMode::LOG_OUTPUT_MODE_UART;
  UART_HandleTypeDef* uart_handle_ = nullptr;

  // Timestamp for log messages
  uint32_t getTimestamp();

  // Level names for printing
  static constexpr std::array<const char*, 5> LEVEL_NAMES = {
    "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"
  };
};

// Convenience macros
#define LOG_DEBUG(...)    TAUV::Logging::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...)     TAUV::Logging::getInstance().info(__VA_ARGS__)
#define LOG_WARN(...)  TAUV::Logging::getInstance().warning(__VA_ARGS__)
#define LOG_ERROR(...)    TAUV::Logging::getInstance().error(__VA_ARGS__)
#define LOG_FATAL(...)    TAUV::Logging::getInstance().fatal(__VA_ARGS__)

} // namespace TAUV

// Global printf function that uses the Logging system
extern "C" int printf(const char* format, ...);
