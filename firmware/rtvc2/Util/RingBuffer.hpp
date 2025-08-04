/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/16/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/
 
#pragma once

namespace TAUV {

template <typename T, size_t Size>
class RingBuffer {
public:
  static_assert((Size & (Size - 1)) == 0, "Size must be a power of 2");

  bool push(T item) {
    size_t next = (head_ + 1) & (Size - 1);
    if (next == tail_) {
      return false; // buffer full
    }
    buffer_[head_] = item;
    head_ = next;
    return true;
  }

  bool pop(T &item) {
    if (tail_ == head_) {
      return false; // buffer empty
    }
    item = buffer_[tail_];
    tail_ = (tail_ + 1) & (Size - 1);
    return true;
  }

  bool isEmpty() const {
    return head_ == tail_;
  }

  bool isFull() const {
    return ((head_ + 1) & (Size - 1)) == tail_;
  }

private:
  T buffer_[Size];
  size_t head_ = 0;
  size_t tail_ = 0;
};

}

