// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tauv_msgs:msg/EscTelemetry.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/esc_telemetry.h"


#ifndef TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__STRUCT_H_
#define TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__STRUCT_H_

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

/// Struct defined in msg/EscTelemetry in the package tauv_msgs.
/**
  * EscTelemetry.msg
 */
typedef struct tauv_msgs__msg__EscTelemetry
{
  std_msgs__msg__Header header;
  /// ESC identifier
  uint8_t id;
  /// Rotations per minute
  int32_t rpm;
  /// Voltage in volts
  float voltage;
  /// Current in amps
  float current;
  /// Temperature in Celsius
  float temperature;
  /// ESC fault code, if any
  uint8_t fault_code;
} tauv_msgs__msg__EscTelemetry;

// Struct for a sequence of tauv_msgs__msg__EscTelemetry.
typedef struct tauv_msgs__msg__EscTelemetry__Sequence
{
  tauv_msgs__msg__EscTelemetry * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tauv_msgs__msg__EscTelemetry__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__STRUCT_H_
