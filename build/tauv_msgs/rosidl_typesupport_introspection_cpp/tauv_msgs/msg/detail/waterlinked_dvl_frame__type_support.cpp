// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__functions.h"
#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__struct.hpp"
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

void WaterlinkedDvlFrame_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) tauv_msgs::msg::WaterlinkedDvlFrame(_init);
}

void WaterlinkedDvlFrame_fini_function(void * message_memory)
{
  auto typed_message = static_cast<tauv_msgs::msg::WaterlinkedDvlFrame *>(message_memory);
  typed_message->~WaterlinkedDvlFrame();
}

size_t size_function__WaterlinkedDvlFrame__covariance(const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * get_const_function__WaterlinkedDvlFrame__covariance(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 9> *>(untyped_member);
  return &member[index];
}

void * get_function__WaterlinkedDvlFrame__covariance(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 9> *>(untyped_member);
  return &member[index];
}

void fetch_function__WaterlinkedDvlFrame__covariance(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__WaterlinkedDvlFrame__covariance(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__WaterlinkedDvlFrame__covariance(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__WaterlinkedDvlFrame__covariance(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__WaterlinkedDvlFrame__transducer_velocity(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__WaterlinkedDvlFrame__transducer_velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__WaterlinkedDvlFrame__transducer_velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__WaterlinkedDvlFrame__transducer_velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__WaterlinkedDvlFrame__transducer_velocity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__WaterlinkedDvlFrame__transducer_velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__WaterlinkedDvlFrame__transducer_velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__WaterlinkedDvlFrame__transducer_distance(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__WaterlinkedDvlFrame__transducer_distance(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__WaterlinkedDvlFrame__transducer_distance(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__WaterlinkedDvlFrame__transducer_distance(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__WaterlinkedDvlFrame__transducer_distance(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__WaterlinkedDvlFrame__transducer_distance(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__WaterlinkedDvlFrame__transducer_distance(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__WaterlinkedDvlFrame__transducer_rssi(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__WaterlinkedDvlFrame__transducer_rssi(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__WaterlinkedDvlFrame__transducer_rssi(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__WaterlinkedDvlFrame__transducer_rssi(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__WaterlinkedDvlFrame__transducer_rssi(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__WaterlinkedDvlFrame__transducer_rssi(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__WaterlinkedDvlFrame__transducer_rssi(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__WaterlinkedDvlFrame__transducer_nsd(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__WaterlinkedDvlFrame__transducer_nsd(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__WaterlinkedDvlFrame__transducer_nsd(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__WaterlinkedDvlFrame__transducer_nsd(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__WaterlinkedDvlFrame__transducer_nsd(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__WaterlinkedDvlFrame__transducer_nsd(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__WaterlinkedDvlFrame__transducer_nsd(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__WaterlinkedDvlFrame__transducer_beam_valid(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__WaterlinkedDvlFrame__transducer_beam_valid(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<bool, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__WaterlinkedDvlFrame__transducer_beam_valid(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<bool, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__WaterlinkedDvlFrame__transducer_beam_valid(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const bool *>(
    get_const_function__WaterlinkedDvlFrame__transducer_beam_valid(untyped_member, index));
  auto & value = *reinterpret_cast<bool *>(untyped_value);
  value = item;
}

void assign_function__WaterlinkedDvlFrame__transducer_beam_valid(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<bool *>(
    get_function__WaterlinkedDvlFrame__transducer_beam_valid(untyped_member, index));
  const auto & value = *reinterpret_cast<const bool *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember WaterlinkedDvlFrame_message_member_array[17] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "time",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, time),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "vx",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, vx),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "vy",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, vy),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "vz",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, vz),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "fom",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, fom),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "covariance",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, covariance),  // bytes offset in struct
    nullptr,  // default value
    size_function__WaterlinkedDvlFrame__covariance,  // size() function pointer
    get_const_function__WaterlinkedDvlFrame__covariance,  // get_const(index) function pointer
    get_function__WaterlinkedDvlFrame__covariance,  // get(index) function pointer
    fetch_function__WaterlinkedDvlFrame__covariance,  // fetch(index, &value) function pointer
    assign_function__WaterlinkedDvlFrame__covariance,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "altitude",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, altitude),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "transducer_velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, transducer_velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__WaterlinkedDvlFrame__transducer_velocity,  // size() function pointer
    get_const_function__WaterlinkedDvlFrame__transducer_velocity,  // get_const(index) function pointer
    get_function__WaterlinkedDvlFrame__transducer_velocity,  // get(index) function pointer
    fetch_function__WaterlinkedDvlFrame__transducer_velocity,  // fetch(index, &value) function pointer
    assign_function__WaterlinkedDvlFrame__transducer_velocity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "transducer_distance",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, transducer_distance),  // bytes offset in struct
    nullptr,  // default value
    size_function__WaterlinkedDvlFrame__transducer_distance,  // size() function pointer
    get_const_function__WaterlinkedDvlFrame__transducer_distance,  // get_const(index) function pointer
    get_function__WaterlinkedDvlFrame__transducer_distance,  // get(index) function pointer
    fetch_function__WaterlinkedDvlFrame__transducer_distance,  // fetch(index, &value) function pointer
    assign_function__WaterlinkedDvlFrame__transducer_distance,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "transducer_rssi",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, transducer_rssi),  // bytes offset in struct
    nullptr,  // default value
    size_function__WaterlinkedDvlFrame__transducer_rssi,  // size() function pointer
    get_const_function__WaterlinkedDvlFrame__transducer_rssi,  // get_const(index) function pointer
    get_function__WaterlinkedDvlFrame__transducer_rssi,  // get(index) function pointer
    fetch_function__WaterlinkedDvlFrame__transducer_rssi,  // fetch(index, &value) function pointer
    assign_function__WaterlinkedDvlFrame__transducer_rssi,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "transducer_nsd",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, transducer_nsd),  // bytes offset in struct
    nullptr,  // default value
    size_function__WaterlinkedDvlFrame__transducer_nsd,  // size() function pointer
    get_const_function__WaterlinkedDvlFrame__transducer_nsd,  // get_const(index) function pointer
    get_function__WaterlinkedDvlFrame__transducer_nsd,  // get(index) function pointer
    fetch_function__WaterlinkedDvlFrame__transducer_nsd,  // fetch(index, &value) function pointer
    assign_function__WaterlinkedDvlFrame__transducer_nsd,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "transducer_beam_valid",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, transducer_beam_valid),  // bytes offset in struct
    nullptr,  // default value
    size_function__WaterlinkedDvlFrame__transducer_beam_valid,  // size() function pointer
    get_const_function__WaterlinkedDvlFrame__transducer_beam_valid,  // get_const(index) function pointer
    get_function__WaterlinkedDvlFrame__transducer_beam_valid,  // get(index) function pointer
    fetch_function__WaterlinkedDvlFrame__transducer_beam_valid,  // fetch(index, &value) function pointer
    assign_function__WaterlinkedDvlFrame__transducer_beam_valid,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "velocity_valid",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, velocity_valid),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "status",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, status),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "time_of_validity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, time_of_validity),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "time_of_transmission",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT64,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(tauv_msgs::msg::WaterlinkedDvlFrame, time_of_transmission),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers WaterlinkedDvlFrame_message_members = {
  "tauv_msgs::msg",  // message namespace
  "WaterlinkedDvlFrame",  // message name
  17,  // number of fields
  sizeof(tauv_msgs::msg::WaterlinkedDvlFrame),
  false,  // has_any_key_member_
  WaterlinkedDvlFrame_message_member_array,  // message members
  WaterlinkedDvlFrame_init_function,  // function to initialize message memory (memory has to be allocated)
  WaterlinkedDvlFrame_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t WaterlinkedDvlFrame_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &WaterlinkedDvlFrame_message_members,
  get_message_typesupport_handle_function,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_hash,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description,
  &tauv_msgs__msg__WaterlinkedDvlFrame__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace tauv_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<tauv_msgs::msg::WaterlinkedDvlFrame>()
{
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::WaterlinkedDvlFrame_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, tauv_msgs, msg, WaterlinkedDvlFrame)() {
  return &::tauv_msgs::msg::rosidl_typesupport_introspection_cpp::WaterlinkedDvlFrame_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
