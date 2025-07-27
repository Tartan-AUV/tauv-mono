// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tauv_msgs:msg/DepthSensorFrame.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/depth_sensor_frame.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__TRAITS_HPP_
#define TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tauv_msgs/msg/detail/depth_sensor_frame__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace tauv_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const DepthSensorFrame & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: depth
  {
    out << "depth: ";
    rosidl_generator_traits::value_to_yaml(msg.depth, out);
    out << ", ";
  }

  // member: pressure
  {
    out << "pressure: ";
    rosidl_generator_traits::value_to_yaml(msg.pressure, out);
    out << ", ";
  }

  // member: temperature
  {
    out << "temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.temperature, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DepthSensorFrame & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: depth
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "depth: ";
    rosidl_generator_traits::value_to_yaml(msg.depth, out);
    out << "\n";
  }

  // member: pressure
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pressure: ";
    rosidl_generator_traits::value_to_yaml(msg.pressure, out);
    out << "\n";
  }

  // member: temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.temperature, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DepthSensorFrame & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace tauv_msgs

namespace rosidl_generator_traits
{

[[deprecated("use tauv_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tauv_msgs::msg::DepthSensorFrame & msg,
  std::ostream & out, size_t indentation = 0)
{
  tauv_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tauv_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const tauv_msgs::msg::DepthSensorFrame & msg)
{
  return tauv_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tauv_msgs::msg::DepthSensorFrame>()
{
  return "tauv_msgs::msg::DepthSensorFrame";
}

template<>
inline const char * name<tauv_msgs::msg::DepthSensorFrame>()
{
  return "tauv_msgs/msg/DepthSensorFrame";
}

template<>
struct has_fixed_size<tauv_msgs::msg::DepthSensorFrame>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<tauv_msgs::msg::DepthSensorFrame>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<tauv_msgs::msg::DepthSensorFrame>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__TRAITS_HPP_
