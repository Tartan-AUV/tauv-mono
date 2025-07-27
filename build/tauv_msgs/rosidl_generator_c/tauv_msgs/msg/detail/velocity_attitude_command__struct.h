// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tauv_msgs:msg/VelocityAttitudeCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/velocity_attitude_command.h"


#ifndef TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__STRUCT_H_
#define TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__STRUCT_H_

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
// Member 'target_velocity'
// Member 'feedforward_acceleration'
#include "geometry_msgs/msg/detail/vector3__struct.h"
// Member 'target_attitude'
#include "geometry_msgs/msg/detail/quaternion__struct.h"

/// Struct defined in msg/VelocityAttitudeCommand in the package tauv_msgs.
/**
  * VelocityAttitudeCommand.msg
  * Command message for specifying desired velocity and attitude for the AUV
 */
typedef struct tauv_msgs__msg__VelocityAttitudeCommand
{
  std_msgs__msg__Header header;
  /// Target linear velocity in body frame
  geometry_msgs__msg__Vector3 target_velocity;
  /// Target attitude as quaternion (orientation in world frame)
  geometry_msgs__msg__Quaternion target_attitude;
  /// Optional feedforward acceleration (if known)
  geometry_msgs__msg__Vector3 feedforward_acceleration;
  /// Control enable flags
  bool velocity_control_enabled;
  bool attitude_control_enabled;
} tauv_msgs__msg__VelocityAttitudeCommand;

// Struct for a sequence of tauv_msgs__msg__VelocityAttitudeCommand.
typedef struct tauv_msgs__msg__VelocityAttitudeCommand__Sequence
{
  tauv_msgs__msg__VelocityAttitudeCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tauv_msgs__msg__VelocityAttitudeCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__STRUCT_H_
