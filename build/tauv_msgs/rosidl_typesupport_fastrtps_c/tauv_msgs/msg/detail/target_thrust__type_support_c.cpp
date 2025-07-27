// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from tauv_msgs:msg/TargetThrust.idl
// generated code does not contain a copyright notice
#include "tauv_msgs/msg/detail/target_thrust__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "tauv_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "tauv_msgs/msg/detail/target_thrust__struct.h"
#include "tauv_msgs/msg/detail/target_thrust__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _TargetThrust__ros_msg_type = tauv_msgs__msg__TargetThrust;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
bool cdr_serialize_tauv_msgs__msg__TargetThrust(
  const tauv_msgs__msg__TargetThrust * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: target_thrust
  {
    size_t size = 8;
    auto array_ptr = ros_message->target_thrust;
    cdr.serialize_array(array_ptr, size);
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
bool cdr_deserialize_tauv_msgs__msg__TargetThrust(
  eprosima::fastcdr::Cdr & cdr,
  tauv_msgs__msg__TargetThrust * ros_message)
{
  // Field name: target_thrust
  {
    size_t size = 8;
    auto array_ptr = ros_message->target_thrust;
    cdr.deserialize_array(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t get_serialized_size_tauv_msgs__msg__TargetThrust(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _TargetThrust__ros_msg_type * ros_message = static_cast<const _TargetThrust__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: target_thrust
  {
    size_t array_size = 8;
    auto array_ptr = ros_message->target_thrust;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t max_serialized_size_tauv_msgs__msg__TargetThrust(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: target_thrust
  {
    size_t array_size = 8;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = tauv_msgs__msg__TargetThrust;
    is_plain =
      (
      offsetof(DataType, target_thrust) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
bool cdr_serialize_key_tauv_msgs__msg__TargetThrust(
  const tauv_msgs__msg__TargetThrust * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: target_thrust
  {
    size_t size = 8;
    auto array_ptr = ros_message->target_thrust;
    cdr.serialize_array(array_ptr, size);
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t get_serialized_size_key_tauv_msgs__msg__TargetThrust(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _TargetThrust__ros_msg_type * ros_message = static_cast<const _TargetThrust__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: target_thrust
  {
    size_t array_size = 8;
    auto array_ptr = ros_message->target_thrust;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t max_serialized_size_key_tauv_msgs__msg__TargetThrust(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;
  // Field name: target_thrust
  {
    size_t array_size = 8;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = tauv_msgs__msg__TargetThrust;
    is_plain =
      (
      offsetof(DataType, target_thrust) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _TargetThrust__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const tauv_msgs__msg__TargetThrust * ros_message = static_cast<const tauv_msgs__msg__TargetThrust *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_tauv_msgs__msg__TargetThrust(ros_message, cdr);
}

static bool _TargetThrust__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  tauv_msgs__msg__TargetThrust * ros_message = static_cast<tauv_msgs__msg__TargetThrust *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_tauv_msgs__msg__TargetThrust(cdr, ros_message);
}

static uint32_t _TargetThrust__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_tauv_msgs__msg__TargetThrust(
      untyped_ros_message, 0));
}

static size_t _TargetThrust__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_tauv_msgs__msg__TargetThrust(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_TargetThrust = {
  "tauv_msgs::msg",
  "TargetThrust",
  _TargetThrust__cdr_serialize,
  _TargetThrust__cdr_deserialize,
  _TargetThrust__get_serialized_size,
  _TargetThrust__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _TargetThrust__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_TargetThrust,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__TargetThrust__get_type_hash,
  &tauv_msgs__msg__TargetThrust__get_type_description,
  &tauv_msgs__msg__TargetThrust__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, tauv_msgs, msg, TargetThrust)() {
  return &_TargetThrust__type_support;
}

#if defined(__cplusplus)
}
#endif
