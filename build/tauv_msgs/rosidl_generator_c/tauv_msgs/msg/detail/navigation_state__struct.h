// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tauv_msgs:msg/NavigationState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/navigation_state.h"


#ifndef TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__STRUCT_H_
#define TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__STRUCT_H_

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
// Member 'body_pose'
#include "geometry_msgs/msg/detail/pose__struct.h"
// Member 'v_b'
// Member 'a_b'
// Member 'omega_b'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/NavigationState in the package tauv_msgs.
/**
  * NavigationState.msg
 */
typedef struct tauv_msgs__msg__NavigationState
{
  std_msgs__msg__Header header;
  geometry_msgs__msg__Pose body_pose;
  geometry_msgs__msg__Vector3 v_b;
  geometry_msgs__msg__Vector3 a_b;
  geometry_msgs__msg__Vector3 omega_b;
} tauv_msgs__msg__NavigationState;

// Struct for a sequence of tauv_msgs__msg__NavigationState.
typedef struct tauv_msgs__msg__NavigationState__Sequence
{
  tauv_msgs__msg__NavigationState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tauv_msgs__msg__NavigationState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__STRUCT_H_
