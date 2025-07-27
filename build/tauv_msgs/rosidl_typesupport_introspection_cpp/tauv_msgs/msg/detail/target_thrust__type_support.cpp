// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from tauv_msgs:msg/TargetThrust.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "tauv_msgs/msg/detail/target_thrust__functions.h"
#include "tauv_msgs/msg/detail/target_thrust__struct.hpp"
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

void TargetThrust_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) tauv_msgs::msg::TargetThrust(_init);
}

void TargetThrust_fini_function(void * message_memory)
{
  auto typed_message = static_cast<tauv_msgs::msg::TargetThrust *>(message_memory);
  typed_message->~TargetThrust();
}

size_t size_function__TargetThrust__target_thrust(const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * get_const_function__TargetThrust__target_thrust(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 8> *>(untyped_member);
  return &member[index];
}

void * get_function__TargetThrust__target_thrust(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 8> *>(untyped_member);
  return &member[index];
}

void fetch_function__TargetThrust__target_thrust(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__TargetThrust__target_thrust(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__TargetThrust__target_thrust(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__TargetThrust__target_thrust(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember TargetThrust_message_member_array[1] = {
  {
    "target_thrust",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::TargetThrust, target_thrust),  // bytes offset in struct
    nullptr,  // default value
    size_function__TargetThrust__target_thrust,  // size() function pointer
    get_const_function__TargetThrust__target_thrust,  // get_const(index) function pointer
    get_function__TargetThrust__target_thrust,  // get(index) function pointer
    fetch_function__TargetThrust__target_thrust,  // fetch(index, &value) function pointer
    assign_function__TargetThrust__target_thrust,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers TargetThrust_message_members = {
  "tauv_msgs::msg",  // message namespace
  "TargetThrust",  // message name
  1,  // number of fields
  sizeof(tauv_msgs::msg::TargetThrust),
  false,  // has_any_key_member_
  TargetThrust_message_member_array,  // message members
  TargetThrust_init_function,  // function to initialize message memory (memory has to be allocated)
  TargetThrust_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t TargetThrust_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &TargetThrust_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__TargetThrust__get_type_hash,
  &tauv_msgs__msg__TargetThrust__get_type_description,
  &tauv_msgs__msg__TargetThrust__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace tauv_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<tauv_msgs::msg::TargetThrust>()
{
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::TargetThrust_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, tauv_msgs, msg, TargetThrust)() {
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::TargetThrust_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
