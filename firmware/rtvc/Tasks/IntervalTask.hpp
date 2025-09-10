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

using std::size_t;

#include "FreeRTOS.h"
#include "timers.h"
#include <atomic>

namespace TAUV {
class IntervalTask {
public:
  virtual ~IntervalTask() = default;

  IntervalTask()
      : timerHandle_(nullptr), running_(false), overlapDetected_(false) {}

  bool start(const char* name, TickType_t periodTicks) {
    timerHandle_ = xTimerCreateStatic(
        name,
        periodTicks,
        pdTRUE,                // Auto-reload
        static_cast<void*>(this),
        &timerCallback,
        &timerStorage_
    );

    if (!timerHandle_) return false;
    return xTimerStart(timerHandle_, 0) == pdPASS;
  }

  bool isOverlapDetected() const { return overlapDetected_.load(); }

protected:
  // Must be implemented by subclass
  virtual void run() = 0;

private:
  TimerHandle_t timerHandle_;
  StaticTimer_t timerStorage_;
  std::atomic<bool> running_;
  std::atomic<bool> overlapDetected_;

  static void timerCallback(TimerHandle_t xTimer) {
    auto* instance = static_cast<IntervalTask*>(pvTimerGetTimerID(xTimer));
    if (instance->running_.exchange(true)) {
      // Task was already running; overlap occurred
      instance->overlapDetected_.store(true);
      return;
    }

    instance->run();
    instance->running_.store(false);
  }
};

}
