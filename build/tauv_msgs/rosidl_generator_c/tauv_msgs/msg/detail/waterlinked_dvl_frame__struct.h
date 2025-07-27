// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/waterlinked_dvl_frame.h"


#ifndef TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__STRUCT_H_
#define TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__STRUCT_H_

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

/// Struct defined in msg/WaterlinkedDvlFrame in the package tauv_msgs.
/**
  * WaterlinkedDvlFrame.msg
 */
typedef struct tauv_msgs__msg__WaterlinkedDvlFrame
{
  std_msgs__msg__Header header;
  /// Timestamp from the DVL device
  double time;
  /// Velocity components in m/s
  double vx;
  double vy;
  double vz;
  /// Figure of merit (lower is better)
  double fom;
  /// 3x3 covariance matrix, flattened row-major
  double covariance[9];
  /// Altitude above seabed in meters
  double altitude;
  /// Transducer-specific measurements (4 beams assumed)
  double transducer_velocity[4];
  double transducer_distance[4];
  double transducer_rssi[4];
  double transducer_nsd[4];
  bool transducer_beam_valid[4];
  /// Whether velocity measurement is valid
  bool velocity_valid;
  /// DVL status code
  int32_t status;
  /// DVL timestamps (e.g. in microseconds since epoch or device boot time)
  int64_t time_of_validity;
  int64_t time_of_transmission;
} tauv_msgs__msg__WaterlinkedDvlFrame;

// Struct for a sequence of tauv_msgs__msg__WaterlinkedDvlFrame.
typedef struct tauv_msgs__msg__WaterlinkedDvlFrame__Sequence
{
  tauv_msgs__msg__WaterlinkedDvlFrame * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tauv_msgs__msg__WaterlinkedDvlFrame__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__STRUCT_H_
