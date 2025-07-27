// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/VelocityAttitudeCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/velocity_attitude_command.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/velocity_attitude_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_VelocityAttitudeCommand_attitude_control_enabled
{
public:
  explicit Init_VelocityAttitudeCommand_attitude_control_enabled(::tauv_msgs::msg::VelocityAttitudeCommand & msg)
  : msg_(msg)
  {}
  ::tauv_msgs::msg::VelocityAttitudeCommand attitude_control_enabled(::tauv_msgs::msg::VelocityAttitudeCommand::_attitude_control_enabled_type arg)
  {
    msg_.attitude_control_enabled = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::VelocityAttitudeCommand msg_;
};

class Init_VelocityAttitudeCommand_velocity_control_enabled
{
public:
  explicit Init_VelocityAttitudeCommand_velocity_control_enabled(::tauv_msgs::msg::VelocityAttitudeCommand & msg)
  : msg_(msg)
  {}
  Init_VelocityAttitudeCommand_attitude_control_enabled velocity_control_enabled(::tauv_msgs::msg::VelocityAttitudeCommand::_velocity_control_enabled_type arg)
  {
    msg_.velocity_control_enabled = std::move(arg);
    return Init_VelocityAttitudeCommand_attitude_control_enabled(msg_);
  }

private:
  ::tauv_msgs::msg::VelocityAttitudeCommand msg_;
};

class Init_VelocityAttitudeCommand_feedforward_acceleration
{
public:
  explicit Init_VelocityAttitudeCommand_feedforward_acceleration(::tauv_msgs::msg::VelocityAttitudeCommand & msg)
  : msg_(msg)
  {}
  Init_VelocityAttitudeCommand_velocity_control_enabled feedforward_acceleration(::tauv_msgs::msg::VelocityAttitudeCommand::_feedforward_acceleration_type arg)
  {
    msg_.feedforward_acceleration = std::move(arg);
    return Init_VelocityAttitudeCommand_velocity_control_enabled(msg_);
  }

private:
  ::tauv_msgs::msg::VelocityAttitudeCommand msg_;
};

class Init_VelocityAttitudeCommand_target_attitude
{
public:
  explicit Init_VelocityAttitudeCommand_target_attitude(::tauv_msgs::msg::VelocityAttitudeCommand & msg)
  : msg_(msg)
  {}
  Init_VelocityAttitudeCommand_feedforward_acceleration target_attitude(::tauv_msgs::msg::VelocityAttitudeCommand::_target_attitude_type arg)
  {
    msg_.target_attitude = std::move(arg);
    return Init_VelocityAttitudeCommand_feedforward_acceleration(msg_);
  }

private:
  ::tauv_msgs::msg::VelocityAttitudeCommand msg_;
};

class Init_VelocityAttitudeCommand_target_velocity
{
public:
  explicit Init_VelocityAttitudeCommand_target_velocity(::tauv_msgs::msg::VelocityAttitudeCommand & msg)
  : msg_(msg)
  {}
  Init_VelocityAttitudeCommand_target_attitude target_velocity(::tauv_msgs::msg::VelocityAttitudeCommand::_target_velocity_type arg)
  {
    msg_.target_velocity = std::move(arg);
    return Init_VelocityAttitudeCommand_target_attitude(msg_);
  }

private:
  ::tauv_msgs::msg::VelocityAttitudeCommand msg_;
};

class Init_VelocityAttitudeCommand_header
{
public:
  Init_VelocityAttitudeCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VelocityAttitudeCommand_target_velocity header(::tauv_msgs::msg::VelocityAttitudeCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_VelocityAttitudeCommand_target_velocity(msg_);
  }

private:
  ::tauv_msgs::msg::VelocityAttitudeCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::VelocityAttitudeCommand>()
{
  return tauv_msgs::msg::builder::Init_VelocityAttitudeCommand_header();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__BUILDER_HPP_
