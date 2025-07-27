// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tauv_msgs/msg/detail/rpm_command__rosidl_typesupport_introspection_c.h"
#include "tauv_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tauv_msgs/msg/detail/rpm_command__functions.h"
#include "tauv_msgs/msg/detail/rpm_command__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tauv_msgs__msg__RpmCommand__init(message_memory);
}

void tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_fini_function(void * message_memory)
{
  tauv_msgs__msg__RpmCommand__fini(message_memory);
}

size_t tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__size_function__RpmCommand__rpms(
  const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_const_function__RpmCommand__rpms(
  const void * untyped_member, size_t index)
{
  const int32_t * member =
    (const int32_t *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_function__RpmCommand__rpms(
  void * untyped_member, size_t index)
{
  int32_t * member =
    (int32_t *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__fetch_function__RpmCommand__rpms(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_const_function__RpmCommand__rpms(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__assign_function__RpmCommand__rpms(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_function__RpmCommand__rpms(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

size_t tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__size_function__RpmCommand__enables(
  const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_const_function__RpmCommand__enables(
  const void * untyped_member, size_t index)
{
  const uint8_t * member =
    (const uint8_t *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_function__RpmCommand__enables(
  void * untyped_member, size_t index)
{
  uint8_t * member =
    (uint8_t *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__fetch_function__RpmCommand__enables(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const uint8_t * item =
    ((const uint8_t *)
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_const_function__RpmCommand__enables(untyped_member, index));
  uint8_t * value =
    (uint8_t *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__assign_function__RpmCommand__enables(
  void * untyped_member, size_t index, const void * untyped_value)
{
  uint8_t * item =
    ((uint8_t *)
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_function__RpmCommand__enables(untyped_member, index));
  const uint8_t * value =
    (const uint8_t *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_member_array[2] = {
  {
    "rpms",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__RpmCommand, rpms),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__size_function__RpmCommand__rpms,  // size() function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_const_function__RpmCommand__rpms,  // get_const(index) function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_function__RpmCommand__rpms,  // get(index) function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__fetch_function__RpmCommand__rpms,  // fetch(index, &value) function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__assign_function__RpmCommand__rpms,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "enables",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__RpmCommand, enables),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__size_function__RpmCommand__enables,  // size() function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_const_function__RpmCommand__enables,  // get_const(index) function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__get_function__RpmCommand__enables,  // get(index) function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__fetch_function__RpmCommand__enables,  // fetch(index, &value) function pointer
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__assign_function__RpmCommand__enables,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_members = {
  "tauv_msgs__msg",  // message namespace
  "RpmCommand",  // message name
  2,  // number of fields
  sizeof(tauv_msgs__msg__RpmCommand),
  false,  // has_any_key_member_
  tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_member_array,  // message members
  tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_type_support_handle = {
  0,
  &tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__RpmCommand__get_type_hash,
  &tauv_msgs__msg__RpmCommand__get_type_description,
  &tauv_msgs__msg__RpmCommand__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tauv_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tauv_msgs, msg, RpmCommand)() {
  if (!tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_type_support_handle.typesupport_identifier) {
    tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tauv_msgs__msg__RpmCommand__rosidl_typesupport_introspection_c__RpmCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
