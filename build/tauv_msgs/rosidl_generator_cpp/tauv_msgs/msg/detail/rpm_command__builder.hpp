// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/rpm_command.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/rpm_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_RpmCommand_enables
{
public:
  explicit Init_RpmCommand_enables(::tauv_msgs::msg::RpmCommand & msg)
  : msg_(msg)
  {}
  ::tauv_msgs::msg::RpmCommand enables(::tauv_msgs::msg::RpmCommand::_enables_type arg)
  {
    msg_.enables = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::RpmCommand msg_;
};

class Init_RpmCommand_rpms
{
public:
  Init_RpmCommand_rpms()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RpmCommand_enables rpms(::tauv_msgs::msg::RpmCommand::_rpms_type arg)
  {
    msg_.rpms = std::move(arg);
    return Init_RpmCommand_enables(msg_);
  }

private:
  ::tauv_msgs::msg::RpmCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::RpmCommand>()
{
  return tauv_msgs::msg::builder::Init_RpmCommand_rpms();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__BUILDER_HPP_
