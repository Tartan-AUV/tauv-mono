/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/
 
#pragma once

extern "C" {
#include "FreeRTOS.h"
#include "queue.h"
}

#include <cstring>
#include <type_traits>

using std::size_t;

template<typename T, size_t QueueLength>
class StaticQueue {
  static_assert(std::is_trivially_copyable<T>::value,
                "StaticQueue only supports trivially copyable types.");

public:
  StaticQueue() {
    handle_ = xQueueCreateStatic(
        QueueLength,
        sizeof(T),
        reinterpret_cast<uint8_t *>(storage_),
        &queueStruct_
    );
    configASSERT(handle_ != nullptr);
  }

  // Delete copy/move to avoid issues
  StaticQueue(const StaticQueue&) = delete;
  StaticQueue& operator=(const StaticQueue&) = delete;

  bool send(const T &item, TickType_t timeout = portMAX_DELAY) {
    return xQueueSend(handle_, &item, timeout) == pdPASS;
  }

  bool sendFromISR(const T &item, BaseType_t *pxHigherPriorityTaskWoken) {
    return xQueueSendFromISR(handle_, &item, pxHigherPriorityTaskWoken) == pdPASS;
  }

  bool receive(T &item, TickType_t timeout = portMAX_DELAY) {
    return xQueueReceive(handle_, &item, timeout) == pdPASS;
  }

  bool receiveFromISR(T &item, BaseType_t *pxHigherPriorityTaskWoken) {
    return xQueueReceiveFromISR(handle_, &item, pxHigherPriorityTaskWoken) == pdPASS;
  }

  bool peek(T &item, TickType_t timeout = portMAX_DELAY) {
    return xQueuePeek(handle_, &item, timeout) == pdPASS;
  }

  size_t messagesWaiting() const {
    return uxQueueMessagesWaiting(handle_);
  }

  QueueHandle_t handle() const {
    return handle_;
  }

private:
  QueueHandle_t handle_;
  StaticQueue_t queueStruct_;
  alignas(T) uint8_t storage_[QueueLength * sizeof(T)];
};
