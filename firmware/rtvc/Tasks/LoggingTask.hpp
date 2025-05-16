/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      FreeRTOS task that processes log messages from the queue and outputs
 *      them to the configured interfaces (UART, ITM/SWDIO, or both).
 *
 *****************************************************************************/
 
#pragma once

#include <memory>
#include "FreeRTOS.h"
#include "task.h"
#include "Logging.hpp"

extern "C" {
#include "cmsis_os.h"
#include "core_cm7.h"
}

namespace TAUV {

class LoggingTask {
public:
    // Constructor
    LoggingTask() = default;
    
    // Destructor
    ~LoggingTask();
    
    // Delete copy constructor and assignment operator
    LoggingTask(const LoggingTask&) = delete;
    LoggingTask& operator=(const LoggingTask&) = delete;
    
    /**
     * Initialize the logging task
     * @return true if initialization was successful
     */
    bool init();
    
    /**
     * Start the task with the given name and priority
     * @param task_name Name to give the task
     * @param priority Task priority (higher number = higher priority)
     * @return true if task was started successfully
     */
    bool start(const char* task_name, uint32_t priority);

private:
    // Handle to the FreeRTOS task
    TaskHandle_t task_handle_ = nullptr;

    // Flag to signal that the task should stop
    volatile bool should_stop_ = false;
    
    // Stack for the task
    static constexpr size_t STACK_SIZE = 512;
    StackType_t task_stack_[STACK_SIZE] = {0};
    StaticTask_t task_buffer_;
    
    // Task entry point - static method that calls the run method
    static void taskEntryPoint(void* params);
    
    // Task main loop
    void run();
    
    // Helper methods for different output modes
    void outputToUART(const LogMessage& msg);
    void outputToITM(const LogMessage& msg);
};

} // namespace TAUV
