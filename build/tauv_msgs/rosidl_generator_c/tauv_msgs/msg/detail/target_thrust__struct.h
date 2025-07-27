// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tauv_msgs:msg/TargetThrust.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/target_thrust.h"


#ifndef TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__STRUCT_H_
#define TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/TargetThrust in the package tauv_msgs.
/**
  * TargetThrust.msg
 */
typedef struct tauv_msgs__msg__TargetThrust
{
  double target_thrust[8];
} tauv_msgs__msg__TargetThrust;

// Struct for a sequence of tauv_msgs__msg__TargetThrust.
typedef struct tauv_msgs__msg__TargetThrust__Sequence
{
  tauv_msgs__msg__TargetThrust * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tauv_msgs__msg__TargetThrust__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__STRUCT_H_
