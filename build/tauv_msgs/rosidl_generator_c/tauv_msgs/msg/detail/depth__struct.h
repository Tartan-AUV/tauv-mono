// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tauv_msgs:msg/Depth.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/depth.h"


#ifndef TAUV_MSGS__MSG__DETAIL__DEPTH__STRUCT_H_
#define TAUV_MSGS__MSG__DETAIL__DEPTH__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/Depth in the package tauv_msgs.
typedef struct tauv_msgs__msg__Depth
{
  std_msgs__msg__Header header;
  /// depth below the surface in meters
  double depth;
  double variance;
} tauv_msgs__msg__Depth;

// Struct for a sequence of tauv_msgs__msg__Depth.
typedef struct tauv_msgs__msg__Depth__Sequence
{
  tauv_msgs__msg__Depth * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tauv_msgs__msg__Depth__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TAUV_MSGS__MSG__DETAIL__DEPTH__STRUCT_H_
