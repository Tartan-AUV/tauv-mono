// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tauv_msgs:msg/NavigationState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tauv_msgs/msg/detail/navigation_state__rosidl_typesupport_introspection_c.h"
#include "tauv_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tauv_msgs/msg/detail/navigation_state__functions.h"
#include "tauv_msgs/msg/detail/navigation_state__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `body_pose`
#include "geometry_msgs/msg/pose.h"
// Member `body_pose`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"
// Member `v_b`
// Member `a_b`
// Member `omega_b`
#include "geometry_msgs/msg/vector3.h"
// Member `v_b`
// Member `a_b`
// Member `omega_b`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tauv_msgs__msg__NavigationState__init(message_memory);
}

void tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_fini_function(void * message_memory)
{
  tauv_msgs__msg__NavigationState__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_member_array[5] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__NavigationState, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "body_pose",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__NavigationState, body_pose),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "v_b",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__NavigationState, v_b),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "a_b",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__NavigationState, a_b),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "omega_b",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__NavigationState, omega_b),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_members = {
  "tauv_msgs__msg",  // message namespace
  "NavigationState",  // message name
  5,  // number of fields
  sizeof(tauv_msgs__msg__NavigationState),
  false,  // has_any_key_member_
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_member_array,  // message members
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_init_function,  // function to initialize message memory (memory has to be allocated)
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_type_support_handle = {
  0,
  &tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__NavigationState__get_type_hash,
  &tauv_msgs__msg__NavigationState__get_type_description,
  &tauv_msgs__msg__NavigationState__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tauv_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tauv_msgs, msg, NavigationState)() {
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  if (!tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_type_support_handle.typesupport_identifier) {
    tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tauv_msgs__msg__NavigationState__rosidl_typesupport_introspection_c__NavigationState_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
