// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tauv_msgs:msg/NavigationState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/navigation_state.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__TRAITS_HPP_
#define TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tauv_msgs/msg/detail/navigation_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'body_pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"
// Member 'v_b'
// Member 'a_b'
// Member 'omega_b'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace tauv_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const NavigationState & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: body_pose
  {
    out << "body_pose: ";
    to_flow_style_yaml(msg.body_pose, out);
    out << ", ";
  }

  // member: v_b
  {
    out << "v_b: ";
    to_flow_style_yaml(msg.v_b, out);
    out << ", ";
  }

  // member: a_b
  {
    out << "a_b: ";
    to_flow_style_yaml(msg.a_b, out);
    out << ", ";
  }

  // member: omega_b
  {
    out << "omega_b: ";
    to_flow_style_yaml(msg.omega_b, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const NavigationState & msg,
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

  // member: body_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "body_pose:\n";
    to_block_style_yaml(msg.body_pose, out, indentation + 2);
  }

  // member: v_b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_b:\n";
    to_block_style_yaml(msg.v_b, out, indentation + 2);
  }

  // member: a_b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "a_b:\n";
    to_block_style_yaml(msg.a_b, out, indentation + 2);
  }

  // member: omega_b
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "omega_b:\n";
    to_block_style_yaml(msg.omega_b, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const NavigationState & msg, bool use_flow_style = false)
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
  const tauv_msgs::msg::NavigationState & msg,
  std::ostream & out, size_t indentation = 0)
{
  tauv_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tauv_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const tauv_msgs::msg::NavigationState & msg)
{
  return tauv_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tauv_msgs::msg::NavigationState>()
{
  return "tauv_msgs::msg::NavigationState";
}

template<>
inline const char * name<tauv_msgs::msg::NavigationState>()
{
  return "tauv_msgs/msg/NavigationState";
}

template<>
struct has_fixed_size<tauv_msgs::msg::NavigationState>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Pose>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<tauv_msgs::msg::NavigationState>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Pose>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<tauv_msgs::msg::NavigationState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__TRAITS_HPP_
