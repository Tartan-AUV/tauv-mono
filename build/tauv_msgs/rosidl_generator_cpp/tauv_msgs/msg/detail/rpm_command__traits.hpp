// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/rpm_command.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__TRAITS_HPP_
#define TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tauv_msgs/msg/detail/rpm_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace tauv_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const RpmCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: rpms
  {
    if (msg.rpms.size() == 0) {
      out << "rpms: []";
    } else {
      out << "rpms: [";
      size_t pending_items = msg.rpms.size();
      for (auto item : msg.rpms) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: enables
  {
    if (msg.enables.size() == 0) {
      out << "enables: []";
    } else {
      out << "enables: [";
      size_t pending_items = msg.enables.size();
      for (auto item : msg.enables) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RpmCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: rpms
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.rpms.size() == 0) {
      out << "rpms: []\n";
    } else {
      out << "rpms:\n";
      for (auto item : msg.rpms) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: enables
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.enables.size() == 0) {
      out << "enables: []\n";
    } else {
      out << "enables:\n";
      for (auto item : msg.enables) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RpmCommand & msg, bool use_flow_style = false)
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
  const tauv_msgs::msg::RpmCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  tauv_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tauv_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const tauv_msgs::msg::RpmCommand & msg)
{
  return tauv_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tauv_msgs::msg::RpmCommand>()
{
  return "tauv_msgs::msg::RpmCommand";
}

template<>
inline const char * name<tauv_msgs::msg::RpmCommand>()
{
  return "tauv_msgs/msg/RpmCommand";
}

template<>
struct has_fixed_size<tauv_msgs::msg::RpmCommand>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<tauv_msgs::msg::RpmCommand>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<tauv_msgs::msg::RpmCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__TRAITS_HPP_
