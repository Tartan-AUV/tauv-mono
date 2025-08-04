/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      Implementation of the LoggingTask class that processes log messages
 *      from the queue and outputs them to the configured interfaces.
 *
 *****************************************************************************/
#include "LoggingTask.hpp"
#include <cstring>

extern "C" {
#include "stm32f7xx_hal.h"
#include "core_cm7.h"
}

namespace TAUV {

LoggingTask::~LoggingTask() {
    // Signal the task to stop and wait for it to exit
    if (task_handle_ != nullptr) {
        should_stop_ = true;
        vTaskDelay(pdMS_TO_TICKS(100)); // Give task time to exit cleanly
        vTaskDelete(task_handle_);
        task_handle_ = nullptr;
    }
}

bool LoggingTask::init() {
    // Nothing specific to initialize for now
    return true;
}

bool LoggingTask::start(const char* task_name, uint32_t priority) {
    if (task_handle_ != nullptr) {
        // Task is already running
        return false;
    }

    should_stop_ = false;
    
    // Create the task
    task_handle_ = xTaskCreateStatic(
        taskEntryPoint,
        task_name,
        STACK_SIZE,
        this,               // Pass this instance as parameter
        priority,
        task_stack_,
        &task_buffer_
    );

    return (task_handle_ != nullptr);
}

void LoggingTask::taskEntryPoint(void* params) {
    // Cast the void pointer back to LoggingTask instance
    auto* task = static_cast<LoggingTask*>(params);
    if (task) {
        task->run();
    }
    
    // Delete the task if we somehow exit the run method
    vTaskDelete(nullptr);
}

void LoggingTask::run() {
    const TickType_t xBlockTime = pdMS_TO_TICKS(100); // Wait up to 100ms for a message
    
    // Get access to the message queue
    QueueHandle_t message_queue = Logging::getMessageQueue();
    if (message_queue == nullptr) {
        return; // Can't do much without a queue
    }
    
    LogMessage msg;
    
    // Main task loop
    while (!should_stop_) {
        // Try to receive a message from the queue
        if (xQueueReceive(message_queue, &msg, xBlockTime) == pdTRUE) {
            // Get current output mode
            LogOutputMode mode = Logging::getInstance().getOutputMode();
            
            // Process the message based on output mode
            if (mode == LogOutputMode::LOG_OUTPUT_MODE_UART || mode == LogOutputMode::LOG_OUTPUT_MODE_ITM_UART) {
                outputToUART(msg);
            }
            
            if (mode == LogOutputMode::LOG_OUTPUT_MODE_ITM || mode == LogOutputMode::LOG_OUTPUT_MODE_ITM_UART) {
                outputToITM(msg);
            }
        }
    }
}

void LoggingTask::outputToUART(const LogMessage& msg) {
    UART_HandleTypeDef* uart = Logging::getInstance().getUartHandle();
    if (uart != nullptr) {
        HAL_UART_Transmit(
            uart,
            reinterpret_cast<const uint8_t*>(msg.buffer),
            msg.length,
            100  // Max timeout 100ms
        );
    }
}

void LoggingTask::outputToITM(const LogMessage& msg) {
    // Send each character through the ITM port
    for (size_t i = 0; i < msg.length; i++) {
        ITM_SendChar(msg.buffer[i]);
    }
}

} // namespace TAUV
