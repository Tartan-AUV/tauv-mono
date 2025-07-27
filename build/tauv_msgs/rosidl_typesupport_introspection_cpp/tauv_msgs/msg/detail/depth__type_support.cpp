// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from tauv_msgs:msg/Depth.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "tauv_msgs/msg/detail/depth__functions.h"
#include "tauv_msgs/msg/detail/depth__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace tauv_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Depth_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) tauv_msgs::msg::Depth(_init);
}

void Depth_fini_function(void * message_memory)
{
  auto typed_message = static_cast<tauv_msgs::msg::Depth *>(message_memory);
  typed_message->~Depth();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Depth_message_member_array[3] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::Depth, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "depth",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::Depth, depth),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "variance",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::Depth, variance),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Depth_message_members = {
  "tauv_msgs::msg",  // message namespace
  "Depth",  // message name
  3,  // number of fields
  sizeof(tauv_msgs::msg::Depth),
  false,  // has_any_key_member_
  Depth_message_member_array,  // message members
  Depth_init_function,  // function to initialize message memory (memory has to be allocated)
  Depth_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Depth_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Depth_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__Depth__get_type_hash,
  &tauv_msgs__msg__Depth__get_type_description,
  &tauv_msgs__msg__Depth__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace tauv_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<tauv_msgs::msg::Depth>()
{
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::Depth_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, tauv_msgs, msg, Depth)() {
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::Depth_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
