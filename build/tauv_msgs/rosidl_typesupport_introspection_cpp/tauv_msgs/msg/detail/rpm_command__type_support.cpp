// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "tauv_msgs/msg/detail/rpm_command__functions.h"
#include "tauv_msgs/msg/detail/rpm_command__struct.hpp"
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

void RpmCommand_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) tauv_msgs::msg::RpmCommand(_init);
}

void RpmCommand_fini_function(void * message_memory)
{
  auto typed_message = static_cast<tauv_msgs::msg::RpmCommand *>(message_memory);
  typed_message->~RpmCommand();
}

size_t size_function__RpmCommand__rpms(const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * get_const_function__RpmCommand__rpms(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<int32_t, 8> *>(untyped_member);
  return &member[index];
}

void * get_function__RpmCommand__rpms(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<int32_t, 8> *>(untyped_member);
  return &member[index];
}

void fetch_function__RpmCommand__rpms(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__RpmCommand__rpms(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__RpmCommand__rpms(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__RpmCommand__rpms(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

size_t size_function__RpmCommand__enables(const void * untyped_member)
{
  (void)untyped_member;
  return 8;
}

const void * get_const_function__RpmCommand__enables(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<uint8_t, 8> *>(untyped_member);
  return &member[index];
}

void * get_function__RpmCommand__enables(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<uint8_t, 8> *>(untyped_member);
  return &member[index];
}

void fetch_function__RpmCommand__enables(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const uint8_t *>(
    get_const_function__RpmCommand__enables(untyped_member, index));
  auto & value = *reinterpret_cast<uint8_t *>(untyped_value);
  value = item;
}

void assign_function__RpmCommand__enables(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<uint8_t *>(
    get_function__RpmCommand__enables(untyped_member, index));
  const auto & value = *reinterpret_cast<const uint8_t *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember RpmCommand_message_member_array[2] = {
  {
    "rpms",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::RpmCommand, rpms),  // bytes offset in struct
    nullptr,  // default value
    size_function__RpmCommand__rpms,  // size() function pointer
    get_const_function__RpmCommand__rpms,  // get_const(index) function pointer
    get_function__RpmCommand__rpms,  // get(index) function pointer
    fetch_function__RpmCommand__rpms,  // fetch(index, &value) function pointer
    assign_function__RpmCommand__rpms,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "enables",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    8,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::RpmCommand, enables),  // bytes offset in struct
    nullptr,  // default value
    size_function__RpmCommand__enables,  // size() function pointer
    get_const_function__RpmCommand__enables,  // get_const(index) function pointer
    get_function__RpmCommand__enables,  // get(index) function pointer
    fetch_function__RpmCommand__enables,  // fetch(index, &value) function pointer
    assign_function__RpmCommand__enables,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers RpmCommand_message_members = {
  "tauv_msgs::msg",  // message namespace
  "RpmCommand",  // message name
  2,  // number of fields
  sizeof(tauv_msgs::msg::RpmCommand),
  false,  // has_any_key_member_
  RpmCommand_message_member_array,  // message members
  RpmCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  RpmCommand_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t RpmCommand_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &RpmCommand_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__RpmCommand__get_type_hash,
  &tauv_msgs__msg__RpmCommand__get_type_description,
  &tauv_msgs__msg__RpmCommand__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace tauv_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<tauv_msgs::msg::RpmCommand>()
{
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::RpmCommand_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, tauv_msgs, msg, RpmCommand)() {
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::RpmCommand_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
