// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__rosidl_typesupport_introspection_c.h"
#include "tauv_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__functions.h"
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tauv_msgs__msg__WaterlinkedDvlFrame__init(message_memory);
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_fini_function(void * message_memory)
{
  tauv_msgs__msg__WaterlinkedDvlFrame__fini(message_memory);
}

size_t tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__covariance(
  const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__covariance(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__covariance(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__covariance(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__covariance(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__covariance(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__covariance(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_velocity(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_velocity(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_velocity(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_velocity(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_velocity(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_distance(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_distance(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_distance(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_distance(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_distance(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_distance(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_distance(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_rssi(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_rssi(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_rssi(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_rssi(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_rssi(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_rssi(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_rssi(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_nsd(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_nsd(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_nsd(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_nsd(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_nsd(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_nsd(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_nsd(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_beam_valid(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_beam_valid(
  const void * untyped_member, size_t index)
{
  const bool * member =
    (const bool *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_beam_valid(
  void * untyped_member, size_t index)
{
  bool * member =
    (bool *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_beam_valid(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const bool * item =
    ((const bool *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_beam_valid(untyped_member, index));
  bool * value =
    (bool *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_beam_valid(
  void * untyped_member, size_t index, const void * untyped_value)
{
  bool * item =
    ((bool *)
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_beam_valid(untyped_member, index));
  const bool * value =
    (const bool *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_member_array[17] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vx",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, vx),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vy",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, vy),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vz",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, vz),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "fom",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, fom),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "covariance",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, covariance),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__covariance,  // size() function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__covariance,  // get_const(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__covariance,  // get(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__covariance,  // fetch(index, &value) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__covariance,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "altitude",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, altitude),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "transducer_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, transducer_velocity),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_velocity,  // size() function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_velocity,  // get_const(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_velocity,  // get(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_velocity,  // fetch(index, &value) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_velocity,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "transducer_distance",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, transducer_distance),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_distance,  // size() function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_distance,  // get_const(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_distance,  // get(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_distance,  // fetch(index, &value) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_distance,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "transducer_rssi",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, transducer_rssi),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_rssi,  // size() function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_rssi,  // get_const(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_rssi,  // get(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_rssi,  // fetch(index, &value) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_rssi,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "transducer_nsd",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, transducer_nsd),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_nsd,  // size() function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_nsd,  // get_const(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_nsd,  // get(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_nsd,  // fetch(index, &value) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_nsd,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "transducer_beam_valid",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, transducer_beam_valid),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__size_function__WaterlinkedDvlFrame__transducer_beam_valid,  // size() function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_const_function__WaterlinkedDvlFrame__transducer_beam_valid,  // get_const(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__get_function__WaterlinkedDvlFrame__transducer_beam_valid,  // get(index) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__fetch_function__WaterlinkedDvlFrame__transducer_beam_valid,  // fetch(index, &value) function pointer
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__assign_function__WaterlinkedDvlFrame__transducer_beam_valid,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "velocity_valid",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, velocity_valid),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "time_of_validity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, time_of_validity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "time_of_transmission",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__WaterlinkedDvlFrame, time_of_transmission),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_members = {
  "tauv_msgs__msg",  // message namespace
  "WaterlinkedDvlFrame",  // message name
  17,  // number of fields
  sizeof(tauv_msgs__msg__WaterlinkedDvlFrame),
  false,  // has_any_key_member_
  tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_member_array,  // message members
  tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_init_function,  // function to initialize message memory (memory has to be allocated)
  tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_type_support_handle = {
  0,
  &tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_hash,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tauv_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tauv_msgs, msg, WaterlinkedDvlFrame)() {
  tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_type_support_handle.typesupport_identifier) {
    tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tauv_msgs__msg__WaterlinkedDvlFrame__rosidl_typesupport_introspection_c__WaterlinkedDvlFrame_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
