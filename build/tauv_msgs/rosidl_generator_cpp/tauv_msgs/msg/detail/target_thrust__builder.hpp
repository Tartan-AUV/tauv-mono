// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/TargetThrust.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/target_thrust.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/target_thrust__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_TargetThrust_target_thrust
{
public:
  Init_TargetThrust_target_thrust()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::tauv_msgs::msg::TargetThrust target_thrust(::tauv_msgs::msg::TargetThrust::_target_thrust_type arg)
  {
    msg_.target_thrust = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::TargetThrust msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::TargetThrust>()
{
  return tauv_msgs::msg::builder::Init_TargetThrust_target_thrust();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__BUILDER_HPP_
