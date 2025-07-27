// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/EscTelemetry.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/esc_telemetry.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/esc_telemetry__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_EscTelemetry_fault_code
{
public:
  explicit Init_EscTelemetry_fault_code(::tauv_msgs::msg::EscTelemetry & msg)
  : msg_(msg)
  {}
  ::tauv_msgs::msg::EscTelemetry fault_code(::tauv_msgs::msg::EscTelemetry::_fault_code_type arg)
  {
    msg_.fault_code = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::EscTelemetry msg_;
};

class Init_EscTelemetry_temperature
{
public:
  explicit Init_EscTelemetry_temperature(::tauv_msgs::msg::EscTelemetry & msg)
  : msg_(msg)
  {}
  Init_EscTelemetry_fault_code temperature(::tauv_msgs::msg::EscTelemetry::_temperature_type arg)
  {
    msg_.temperature = std::move(arg);
    return Init_EscTelemetry_fault_code(msg_);
  }

private:
  ::tauv_msgs::msg::EscTelemetry msg_;
};

class Init_EscTelemetry_current
{
public:
  explicit Init_EscTelemetry_current(::tauv_msgs::msg::EscTelemetry & msg)
  : msg_(msg)
  {}
  Init_EscTelemetry_temperature current(::tauv_msgs::msg::EscTelemetry::_current_type arg)
  {
    msg_.current = std::move(arg);
    return Init_EscTelemetry_temperature(msg_);
  }

private:
  ::tauv_msgs::msg::EscTelemetry msg_;
};

class Init_EscTelemetry_voltage
{
public:
  explicit Init_EscTelemetry_voltage(::tauv_msgs::msg::EscTelemetry & msg)
  : msg_(msg)
  {}
  Init_EscTelemetry_current voltage(::tauv_msgs::msg::EscTelemetry::_voltage_type arg)
  {
    msg_.voltage = std::move(arg);
    return Init_EscTelemetry_current(msg_);
  }

private:
  ::tauv_msgs::msg::EscTelemetry msg_;
};

class Init_EscTelemetry_rpm
{
public:
  explicit Init_EscTelemetry_rpm(::tauv_msgs::msg::EscTelemetry & msg)
  : msg_(msg)
  {}
  Init_EscTelemetry_voltage rpm(::tauv_msgs::msg::EscTelemetry::_rpm_type arg)
  {
    msg_.rpm = std::move(arg);
    return Init_EscTelemetry_voltage(msg_);
  }

private:
  ::tauv_msgs::msg::EscTelemetry msg_;
};

class Init_EscTelemetry_id
{
public:
  explicit Init_EscTelemetry_id(::tauv_msgs::msg::EscTelemetry & msg)
  : msg_(msg)
  {}
  Init_EscTelemetry_rpm id(::tauv_msgs::msg::EscTelemetry::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_EscTelemetry_rpm(msg_);
  }

private:
  ::tauv_msgs::msg::EscTelemetry msg_;
};

class Init_EscTelemetry_header
{
public:
  Init_EscTelemetry_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EscTelemetry_id header(::tauv_msgs::msg::EscTelemetry::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_EscTelemetry_id(msg_);
  }

private:
  ::tauv_msgs::msg::EscTelemetry msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::EscTelemetry>()
{
  return tauv_msgs::msg::builder::Init_EscTelemetry_header();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__BUILDER_HPP_
