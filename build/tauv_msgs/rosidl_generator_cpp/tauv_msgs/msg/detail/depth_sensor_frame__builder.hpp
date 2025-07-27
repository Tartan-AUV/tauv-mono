// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/DepthSensorFrame.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/depth_sensor_frame.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/depth_sensor_frame__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_DepthSensorFrame_temperature
{
public:
  explicit Init_DepthSensorFrame_temperature(::tauv_msgs::msg::DepthSensorFrame & msg)
  : msg_(msg)
  {}
  ::tauv_msgs::msg::DepthSensorFrame temperature(::tauv_msgs::msg::DepthSensorFrame::_temperature_type arg)
  {
    msg_.temperature = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::DepthSensorFrame msg_;
};

class Init_DepthSensorFrame_pressure
{
public:
  explicit Init_DepthSensorFrame_pressure(::tauv_msgs::msg::DepthSensorFrame & msg)
  : msg_(msg)
  {}
  Init_DepthSensorFrame_temperature pressure(::tauv_msgs::msg::DepthSensorFrame::_pressure_type arg)
  {
    msg_.pressure = std::move(arg);
    return Init_DepthSensorFrame_temperature(msg_);
  }

private:
  ::tauv_msgs::msg::DepthSensorFrame msg_;
};

class Init_DepthSensorFrame_depth
{
public:
  explicit Init_DepthSensorFrame_depth(::tauv_msgs::msg::DepthSensorFrame & msg)
  : msg_(msg)
  {}
  Init_DepthSensorFrame_pressure depth(::tauv_msgs::msg::DepthSensorFrame::_depth_type arg)
  {
    msg_.depth = std::move(arg);
    return Init_DepthSensorFrame_pressure(msg_);
  }

private:
  ::tauv_msgs::msg::DepthSensorFrame msg_;
};

class Init_DepthSensorFrame_header
{
public:
  Init_DepthSensorFrame_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DepthSensorFrame_depth header(::tauv_msgs::msg::DepthSensorFrame::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_DepthSensorFrame_depth(msg_);
  }

private:
  ::tauv_msgs::msg::DepthSensorFrame msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::DepthSensorFrame>()
{
  return tauv_msgs::msg::builder::Init_DepthSensorFrame_header();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__BUILDER_HPP_
