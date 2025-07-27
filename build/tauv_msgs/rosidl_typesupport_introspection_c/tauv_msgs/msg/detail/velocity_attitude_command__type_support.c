// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from tauv_msgs:msg/VelocityAttitudeCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "tauv_msgs/msg/detail/velocity_attitude_command__rosidl_typesupport_introspection_c.h"
#include "tauv_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "tauv_msgs/msg/detail/velocity_attitude_command__functions.h"
#include "tauv_msgs/msg/detail/velocity_attitude_command__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `target_velocity`
// Member `feedforward_acceleration`
#include "geometry_msgs/msg/vector3.h"
// Member `target_velocity`
// Member `feedforward_acceleration`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"
// Member `target_attitude`
#include "geometry_msgs/msg/quaternion.h"
// Member `target_attitude`
#include "geometry_msgs/msg/detail/quaternion__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  tauv_msgs__msg__VelocityAttitudeCommand__init(message_memory);
}

void tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_fini_function(void * message_memory)
{
  tauv_msgs__msg__VelocityAttitudeCommand__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__VelocityAttitudeCommand, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__VelocityAttitudeCommand, target_velocity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_attitude",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__VelocityAttitudeCommand, target_attitude),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "feedforward_acceleration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__VelocityAttitudeCommand, feedforward_acceleration),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "velocity_control_enabled",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__VelocityAttitudeCommand, velocity_control_enabled),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "attitude_control_enabled",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs__msg__VelocityAttitudeCommand, attitude_control_enabled),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_members = {
  "tauv_msgs__msg",  // message namespace
  "VelocityAttitudeCommand",  // message name
  6,  // number of fields
  sizeof(tauv_msgs__msg__VelocityAttitudeCommand),
  false,  // has_any_key_member_
  tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_member_array,  // message members
  tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_type_support_handle = {
  0,
  &tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__VelocityAttitudeCommand__get_type_hash,
  &tauv_msgs__msg__VelocityAttitudeCommand__get_type_description,
  &tauv_msgs__msg__VelocityAttitudeCommand__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_tauv_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, tauv_msgs, msg, VelocityAttitudeCommand)() {
  tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Quaternion)();
  tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  if (!tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_type_support_handle.typesupport_identifier) {
    tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &tauv_msgs__msg__VelocityAttitudeCommand__rosidl_typesupport_introspection_c__VelocityAttitudeCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
