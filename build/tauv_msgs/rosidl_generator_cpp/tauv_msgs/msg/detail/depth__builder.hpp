// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/Depth.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/depth.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__DEPTH__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__DEPTH__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/depth__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_Depth_variance
{
public:
  explicit Init_Depth_variance(::tauv_msgs::msg::Depth & msg)
  : msg_(msg)
  {}
  ::tauv_msgs::msg::Depth variance(::tauv_msgs::msg::Depth::_variance_type arg)
  {
    msg_.variance = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::Depth msg_;
};

class Init_Depth_depth
{
public:
  explicit Init_Depth_depth(::tauv_msgs::msg::Depth & msg)
  : msg_(msg)
  {}
  Init_Depth_variance depth(::tauv_msgs::msg::Depth::_depth_type arg)
  {
    msg_.depth = std::move(arg);
    return Init_Depth_variance(msg_);
  }

private:
  ::tauv_msgs::msg::Depth msg_;
};

class Init_Depth_header
{
public:
  Init_Depth_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Depth_depth header(::tauv_msgs::msg::Depth::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Depth_depth(msg_);
  }

private:
  ::tauv_msgs::msg::Depth msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::Depth>()
{
  return tauv_msgs::msg::builder::Init_Depth_header();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__DEPTH__BUILDER_HPP_
