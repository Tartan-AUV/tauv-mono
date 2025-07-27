// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/rpm_command.h"


#ifndef TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__STRUCT_H_
#define TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/RpmCommand in the package tauv_msgs.
/**
  * RpmCommand.msg
 */
typedef struct tauv_msgs__msg__RpmCommand
{
  int32_t rpms[8];
  uint8_t enables[8];
} tauv_msgs__msg__RpmCommand;

// Struct for a sequence of tauv_msgs__msg__RpmCommand.
typedef struct tauv_msgs__msg__RpmCommand__Sequence
{
  tauv_msgs__msg__RpmCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tauv_msgs__msg__RpmCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__STRUCT_H_
