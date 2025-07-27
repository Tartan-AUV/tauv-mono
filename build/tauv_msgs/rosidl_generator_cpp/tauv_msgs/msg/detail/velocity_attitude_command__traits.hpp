// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tauv_msgs:msg/VelocityAttitudeCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/velocity_attitude_command.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__TRAITS_HPP_
#define TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tauv_msgs/msg/detail/velocity_attitude_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'target_velocity'
// Member 'feedforward_acceleration'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"
// Member 'target_attitude'
#include "geometry_msgs/msg/detail/quaternion__traits.hpp"

namespace tauv_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const VelocityAttitudeCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: target_velocity
  {
    out << "target_velocity: ";
    to_flow_style_yaml(msg.target_velocity, out);
    out << ", ";
  }

  // member: target_attitude
  {
    out << "target_attitude: ";
    to_flow_style_yaml(msg.target_attitude, out);
    out << ", ";
  }

  // member: feedforward_acceleration
  {
    out << "feedforward_acceleration: ";
    to_flow_style_yaml(msg.feedforward_acceleration, out);
    out << ", ";
  }

  // member: velocity_control_enabled
  {
    out << "velocity_control_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity_control_enabled, out);
    out << ", ";
  }

  // member: attitude_control_enabled
  {
    out << "attitude_control_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.attitude_control_enabled, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const VelocityAttitudeCommand & msg,
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

  // member: target_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_velocity:\n";
    to_block_style_yaml(msg.target_velocity, out, indentation + 2);
  }

  // member: target_attitude
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_attitude:\n";
    to_block_style_yaml(msg.target_attitude, out, indentation + 2);
  }

  // member: feedforward_acceleration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "feedforward_acceleration:\n";
    to_block_style_yaml(msg.feedforward_acceleration, out, indentation + 2);
  }

  // member: velocity_control_enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "velocity_control_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity_control_enabled, out);
    out << "\n";
  }

  // member: attitude_control_enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "attitude_control_enabled: ";
    rosidl_generator_traits::value_to_yaml(msg.attitude_control_enabled, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const VelocityAttitudeCommand & msg, bool use_flow_style = false)
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
  const tauv_msgs::msg::VelocityAttitudeCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  tauv_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tauv_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const tauv_msgs::msg::VelocityAttitudeCommand & msg)
{
  return tauv_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tauv_msgs::msg::VelocityAttitudeCommand>()
{
  return "tauv_msgs::msg::VelocityAttitudeCommand";
}

template<>
inline const char * name<tauv_msgs::msg::VelocityAttitudeCommand>()
{
  return "tauv_msgs/msg/VelocityAttitudeCommand";
}

template<>
struct has_fixed_size<tauv_msgs::msg::VelocityAttitudeCommand>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Quaternion>::value && has_fixed_size<geometry_msgs::msg::Vector3>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<tauv_msgs::msg::VelocityAttitudeCommand>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Quaternion>::value && has_bounded_size<geometry_msgs::msg::Vector3>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<tauv_msgs::msg::VelocityAttitudeCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__TRAITS_HPP_
