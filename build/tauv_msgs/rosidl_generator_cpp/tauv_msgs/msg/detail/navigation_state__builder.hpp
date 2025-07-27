// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/NavigationState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/navigation_state.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/navigation_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_NavigationState_omega_b
{
public:
  explicit Init_NavigationState_omega_b(::tauv_msgs::msg::NavigationState & msg)
  : msg_(msg)
  {}
  ::tauv_msgs::msg::NavigationState omega_b(::tauv_msgs::msg::NavigationState::_omega_b_type arg)
  {
    msg_.omega_b = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::NavigationState msg_;
};

class Init_NavigationState_a_b
{
public:
  explicit Init_NavigationState_a_b(::tauv_msgs::msg::NavigationState & msg)
  : msg_(msg)
  {}
  Init_NavigationState_omega_b a_b(::tauv_msgs::msg::NavigationState::_a_b_type arg)
  {
    msg_.a_b = std::move(arg);
    return Init_NavigationState_omega_b(msg_);
  }

private:
  ::tauv_msgs::msg::NavigationState msg_;
};

class Init_NavigationState_v_b
{
public:
  explicit Init_NavigationState_v_b(::tauv_msgs::msg::NavigationState & msg)
  : msg_(msg)
  {}
  Init_NavigationState_a_b v_b(::tauv_msgs::msg::NavigationState::_v_b_type arg)
  {
    msg_.v_b = std::move(arg);
    return Init_NavigationState_a_b(msg_);
  }

private:
  ::tauv_msgs::msg::NavigationState msg_;
};

class Init_NavigationState_body_pose
{
public:
  explicit Init_NavigationState_body_pose(::tauv_msgs::msg::NavigationState & msg)
  : msg_(msg)
  {}
  Init_NavigationState_v_b body_pose(::tauv_msgs::msg::NavigationState::_body_pose_type arg)
  {
    msg_.body_pose = std::move(arg);
    return Init_NavigationState_v_b(msg_);
  }

private:
  ::tauv_msgs::msg::NavigationState msg_;
};

class Init_NavigationState_header
{
public:
  Init_NavigationState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NavigationState_body_pose header(::tauv_msgs::msg::NavigationState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_NavigationState_body_pose(msg_);
  }

private:
  ::tauv_msgs::msg::NavigationState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::NavigationState>()
{
  return tauv_msgs::msg::builder::Init_NavigationState_header();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__BUILDER_HPP_
