/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/15/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/
 
#pragma once

#include <cstddef>

#include "FreeRTOS.h"
#include "TaskBase.h"

using std::size_t;

namespace TAUV {

class Task {
public:
  virtual ~Task() = default;

  // Delete copy and move
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;
  Task(Task&&) = delete;
  Task& operator=(Task&&) = delete;

  // Start the task
  bool start(const char* name,
             uint16_t stackSize,
             UBaseType_t priority,
             BaseType_t coreAffinity = tskNO_AFFINITY) {
    BaseType_t result = xTaskCreatePinnedToCore(
        taskEntry, name, stackSize,
        this, priority,
        &taskHandle_,
        coreAffinity
    );

    return result == pdPASS;
  }

  TaskHandle_t getHandle() const { return taskHandle_; }

protected:
  Task() : taskHandle_(nullptr) {}

  // This must be implemented by derived class
  virtual void run() = 0;

private:
  TaskHandle_t taskHandle_;

  static void taskEntry(void* pvParameters) {
    auto* instance = static_cast<Task*>(pvParameters);
    instance->run();
    vTaskDelete(nullptr);  // Clean up when run() exits
  }
};

}

