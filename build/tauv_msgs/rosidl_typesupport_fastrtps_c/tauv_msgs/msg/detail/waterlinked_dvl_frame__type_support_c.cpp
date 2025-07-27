// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "tauv_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__struct.h"
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__functions.h"
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

#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
bool cdr_serialize_std_msgs__msg__Header(
  const std_msgs__msg__Header * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
bool cdr_deserialize_std_msgs__msg__Header(
  eprosima::fastcdr::Cdr & cdr,
  std_msgs__msg__Header * ros_message);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
size_t get_serialized_size_std_msgs__msg__Header(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
size_t max_serialized_size_std_msgs__msg__Header(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
bool cdr_serialize_key_std_msgs__msg__Header(
  const std_msgs__msg__Header * ros_message,
  eprosima::fastcdr::Cdr & cdr);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
size_t get_serialized_size_key_std_msgs__msg__Header(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
size_t max_serialized_size_key_std_msgs__msg__Header(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_tauv_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, std_msgs, msg, Header)();


using _WaterlinkedDvlFrame__ros_msg_type = tauv_msgs__msg__WaterlinkedDvlFrame;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
bool cdr_serialize_tauv_msgs__msg__WaterlinkedDvlFrame(
  const tauv_msgs__msg__WaterlinkedDvlFrame * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: header
  {
    cdr_serialize_std_msgs__msg__Header(
      &ros_message->header, cdr);
  }

  // Field name: time
  {
    cdr << ros_message->time;
  }

  // Field name: vx
  {
    cdr << ros_message->vx;
  }

  // Field name: vy
  {
    cdr << ros_message->vy;
  }

  // Field name: vz
  {
    cdr << ros_message->vz;
  }

  // Field name: fom
  {
    cdr << ros_message->fom;
  }

  // Field name: covariance
  {
    size_t size = 9;
    auto array_ptr = ros_message->covariance;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: altitude
  {
    cdr << ros_message->altitude;
  }

  // Field name: transducer_velocity
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_velocity;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_distance
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_distance;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_rssi
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_rssi;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_nsd
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_nsd;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_beam_valid
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_beam_valid;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: velocity_valid
  {
    cdr << (ros_message->velocity_valid ? true : false);
  }

  // Field name: status
  {
    cdr << ros_message->status;
  }

  // Field name: time_of_validity
  {
    cdr << ros_message->time_of_validity;
  }

  // Field name: time_of_transmission
  {
    cdr << ros_message->time_of_transmission;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
bool cdr_deserialize_tauv_msgs__msg__WaterlinkedDvlFrame(
  eprosima::fastcdr::Cdr & cdr,
  tauv_msgs__msg__WaterlinkedDvlFrame * ros_message)
{
  // Field name: header
  {
    cdr_deserialize_std_msgs__msg__Header(cdr, &ros_message->header);
  }

  // Field name: time
  {
    cdr >> ros_message->time;
  }

  // Field name: vx
  {
    cdr >> ros_message->vx;
  }

  // Field name: vy
  {
    cdr >> ros_message->vy;
  }

  // Field name: vz
  {
    cdr >> ros_message->vz;
  }

  // Field name: fom
  {
    cdr >> ros_message->fom;
  }

  // Field name: covariance
  {
    size_t size = 9;
    auto array_ptr = ros_message->covariance;
    cdr.deserialize_array(array_ptr, size);
  }

  // Field name: altitude
  {
    cdr >> ros_message->altitude;
  }

  // Field name: transducer_velocity
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_velocity;
    cdr.deserialize_array(array_ptr, size);
  }

  // Field name: transducer_distance
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_distance;
    cdr.deserialize_array(array_ptr, size);
  }

  // Field name: transducer_rssi
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_rssi;
    cdr.deserialize_array(array_ptr, size);
  }

  // Field name: transducer_nsd
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_nsd;
    cdr.deserialize_array(array_ptr, size);
  }

  // Field name: transducer_beam_valid
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_beam_valid;
    for (size_t i = 0; i < size; ++i) {
      uint8_t tmp;
      cdr >> tmp;
      array_ptr[i] = tmp ? true : false;
    }
  }

  // Field name: velocity_valid
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->velocity_valid = tmp ? true : false;
  }

  // Field name: status
  {
    cdr >> ros_message->status;
  }

  // Field name: time_of_validity
  {
    cdr >> ros_message->time_of_validity;
  }

  // Field name: time_of_transmission
  {
    cdr >> ros_message->time_of_transmission;
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t get_serialized_size_tauv_msgs__msg__WaterlinkedDvlFrame(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _WaterlinkedDvlFrame__ros_msg_type * ros_message = static_cast<const _WaterlinkedDvlFrame__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: header
  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);

  // Field name: time
  {
    size_t item_size = sizeof(ros_message->time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: vx
  {
    size_t item_size = sizeof(ros_message->vx);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: vy
  {
    size_t item_size = sizeof(ros_message->vy);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: vz
  {
    size_t item_size = sizeof(ros_message->vz);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: fom
  {
    size_t item_size = sizeof(ros_message->fom);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: covariance
  {
    size_t array_size = 9;
    auto array_ptr = ros_message->covariance;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: altitude
  {
    size_t item_size = sizeof(ros_message->altitude);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_velocity
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_distance
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_distance;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_rssi
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_rssi;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_nsd
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_nsd;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_beam_valid
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_beam_valid;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: velocity_valid
  {
    size_t item_size = sizeof(ros_message->velocity_valid);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: status
  {
    size_t item_size = sizeof(ros_message->status);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: time_of_validity
  {
    size_t item_size = sizeof(ros_message->time_of_validity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: time_of_transmission
  {
    size_t item_size = sizeof(ros_message->time_of_transmission);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t max_serialized_size_tauv_msgs__msg__WaterlinkedDvlFrame(
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

  // Field name: header
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_std_msgs__msg__Header(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: time
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: vx
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: vy
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: vz
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: fom
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: covariance
  {
    size_t array_size = 9;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: altitude
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_velocity
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_distance
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_rssi
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_nsd
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_beam_valid
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Field name: velocity_valid
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Field name: status
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: time_of_validity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: time_of_transmission
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = tauv_msgs__msg__WaterlinkedDvlFrame;
    is_plain =
      (
      offsetof(DataType, time_of_transmission) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
bool cdr_serialize_key_tauv_msgs__msg__WaterlinkedDvlFrame(
  const tauv_msgs__msg__WaterlinkedDvlFrame * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: header
  {
    cdr_serialize_key_std_msgs__msg__Header(
      &ros_message->header, cdr);
  }

  // Field name: time
  {
    cdr << ros_message->time;
  }

  // Field name: vx
  {
    cdr << ros_message->vx;
  }

  // Field name: vy
  {
    cdr << ros_message->vy;
  }

  // Field name: vz
  {
    cdr << ros_message->vz;
  }

  // Field name: fom
  {
    cdr << ros_message->fom;
  }

  // Field name: covariance
  {
    size_t size = 9;
    auto array_ptr = ros_message->covariance;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: altitude
  {
    cdr << ros_message->altitude;
  }

  // Field name: transducer_velocity
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_velocity;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_distance
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_distance;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_rssi
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_rssi;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_nsd
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_nsd;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: transducer_beam_valid
  {
    size_t size = 4;
    auto array_ptr = ros_message->transducer_beam_valid;
    cdr.serialize_array(array_ptr, size);
  }

  // Field name: velocity_valid
  {
    cdr << (ros_message->velocity_valid ? true : false);
  }

  // Field name: status
  {
    cdr << ros_message->status;
  }

  // Field name: time_of_validity
  {
    cdr << ros_message->time_of_validity;
  }

  // Field name: time_of_transmission
  {
    cdr << ros_message->time_of_transmission;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t get_serialized_size_key_tauv_msgs__msg__WaterlinkedDvlFrame(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _WaterlinkedDvlFrame__ros_msg_type * ros_message = static_cast<const _WaterlinkedDvlFrame__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: header
  current_alignment += get_serialized_size_key_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);

  // Field name: time
  {
    size_t item_size = sizeof(ros_message->time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: vx
  {
    size_t item_size = sizeof(ros_message->vx);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: vy
  {
    size_t item_size = sizeof(ros_message->vy);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: vz
  {
    size_t item_size = sizeof(ros_message->vz);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: fom
  {
    size_t item_size = sizeof(ros_message->fom);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: covariance
  {
    size_t array_size = 9;
    auto array_ptr = ros_message->covariance;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: altitude
  {
    size_t item_size = sizeof(ros_message->altitude);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_velocity
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_distance
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_distance;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_rssi
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_rssi;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_nsd
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_nsd;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: transducer_beam_valid
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->transducer_beam_valid;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: velocity_valid
  {
    size_t item_size = sizeof(ros_message->velocity_valid);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: status
  {
    size_t item_size = sizeof(ros_message->status);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: time_of_validity
  {
    size_t item_size = sizeof(ros_message->time_of_validity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: time_of_transmission
  {
    size_t item_size = sizeof(ros_message->time_of_transmission);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_tauv_msgs
size_t max_serialized_size_key_tauv_msgs__msg__WaterlinkedDvlFrame(
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
  // Field name: header
  {
    size_t array_size = 1;
    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_key_std_msgs__msg__Header(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Field name: time
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: vx
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: vy
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: vz
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: fom
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: covariance
  {
    size_t array_size = 9;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: altitude
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_velocity
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_distance
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_rssi
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_nsd
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: transducer_beam_valid
  {
    size_t array_size = 4;
    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Field name: velocity_valid
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Field name: status
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: time_of_validity
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: time_of_transmission
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = tauv_msgs__msg__WaterlinkedDvlFrame;
    is_plain =
      (
      offsetof(DataType, time_of_transmission) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _WaterlinkedDvlFrame__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const tauv_msgs__msg__WaterlinkedDvlFrame * ros_message = static_cast<const tauv_msgs__msg__WaterlinkedDvlFrame *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_tauv_msgs__msg__WaterlinkedDvlFrame(ros_message, cdr);
}

static bool _WaterlinkedDvlFrame__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  tauv_msgs__msg__WaterlinkedDvlFrame * ros_message = static_cast<tauv_msgs__msg__WaterlinkedDvlFrame *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_tauv_msgs__msg__WaterlinkedDvlFrame(cdr, ros_message);
}

static uint32_t _WaterlinkedDvlFrame__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_tauv_msgs__msg__WaterlinkedDvlFrame(
      untyped_ros_message, 0));
}

static size_t _WaterlinkedDvlFrame__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_tauv_msgs__msg__WaterlinkedDvlFrame(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_WaterlinkedDvlFrame = {
  "tauv_msgs::msg",
  "WaterlinkedDvlFrame",
  _WaterlinkedDvlFrame__cdr_serialize,
  _WaterlinkedDvlFrame__cdr_deserialize,
  _WaterlinkedDvlFrame__get_serialized_size,
  _WaterlinkedDvlFrame__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _WaterlinkedDvlFrame__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_WaterlinkedDvlFrame,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_hash,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, tauv_msgs, msg, WaterlinkedDvlFrame)() {
  return &_WaterlinkedDvlFrame__type_support;
}

#if defined(__cplusplus)
}
#endif
