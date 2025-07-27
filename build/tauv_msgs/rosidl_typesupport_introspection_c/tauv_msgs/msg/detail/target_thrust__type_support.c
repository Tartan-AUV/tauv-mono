// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tauv_msgs:msg/TargetThrust.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tauv_msgs/msg/detail/target_thrust__rosidl_typesupport_introspection_c.h"
#include "tauv_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tauv_msgs/msg/detail/target_thrust__functions.h"
#include "tauv_msgs/msg/detail/target_thrust__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tauv_msgs__msg__TargetThrust__init(message_memory);
}

void tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_fini_function(void * message_memory)
{
  tauv_msgs__msg__TargetThrust__fini(message_memory);
}

size_t tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__size_function__TargetThrust__target_thrust(
  const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__get_const_function__TargetThrust__target_thrust(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__get_function__TargetThrust__target_thrust(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__fetch_function__TargetThrust__target_thrust(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__get_const_function__TargetThrust__target_thrust(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__assign_function__TargetThrust__target_thrust(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__get_function__TargetThrust__target_thrust(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_member_array[1] = {
  {
    "target_thrust",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__TargetThrust, target_thrust),  // bytes offset in struct
    NULL,  // default value
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__size_function__TargetThrust__target_thrust,  // size() function pointer
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__get_const_function__TargetThrust__target_thrust,  // get_const(index) function pointer
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__get_function__TargetThrust__target_thrust,  // get(index) function pointer
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__fetch_function__TargetThrust__target_thrust,  // fetch(index, &value) function pointer
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__assign_function__TargetThrust__target_thrust,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_members = {
  "tauv_msgs__msg",  // message namespace
  "TargetThrust",  // message name
  1,  // number of fields
  sizeof(tauv_msgs__msg__TargetThrust),
  false,  // has_any_key_member_
  tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_member_array,  // message members
  tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_init_function,  // function to initialize message memory (memory has to be allocated)
  tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_type_support_handle = {
  0,
  &tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__TargetThrust__get_type_hash,
  &tauv_msgs__msg__TargetThrust__get_type_description,
  &tauv_msgs__msg__TargetThrust__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tauv_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tauv_msgs, msg, TargetThrust)() {
  if (!tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_type_support_handle.typesupport_identifier) {
    tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tauv_msgs__msg__TargetThrust__rosidl_typesupport_introspection_c__TargetThrust_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
