// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/waterlinked_dvl_frame.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__BUILDER_HPP_
#define TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tauv_msgs
{

namespace msg
{

namespace builder
{

class Init_WaterlinkedDvlFrame_time_of_transmission
{
public:
  explicit Init_WaterlinkedDvlFrame_time_of_transmission(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  ::tauv_msgs::msg::WaterlinkedDvlFrame time_of_transmission(::tauv_msgs::msg::WaterlinkedDvlFrame::_time_of_transmission_type arg)
  {
    msg_.time_of_transmission = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_time_of_validity
{
public:
  explicit Init_WaterlinkedDvlFrame_time_of_validity(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_time_of_transmission time_of_validity(::tauv_msgs::msg::WaterlinkedDvlFrame::_time_of_validity_type arg)
  {
    msg_.time_of_validity = std::move(arg);
    return Init_WaterlinkedDvlFrame_time_of_transmission(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_status
{
public:
  explicit Init_WaterlinkedDvlFrame_status(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_time_of_validity status(::tauv_msgs::msg::WaterlinkedDvlFrame::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_WaterlinkedDvlFrame_time_of_validity(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_velocity_valid
{
public:
  explicit Init_WaterlinkedDvlFrame_velocity_valid(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_status velocity_valid(::tauv_msgs::msg::WaterlinkedDvlFrame::_velocity_valid_type arg)
  {
    msg_.velocity_valid = std::move(arg);
    return Init_WaterlinkedDvlFrame_status(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_transducer_beam_valid
{
public:
  explicit Init_WaterlinkedDvlFrame_transducer_beam_valid(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_velocity_valid transducer_beam_valid(::tauv_msgs::msg::WaterlinkedDvlFrame::_transducer_beam_valid_type arg)
  {
    msg_.transducer_beam_valid = std::move(arg);
    return Init_WaterlinkedDvlFrame_velocity_valid(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_transducer_nsd
{
public:
  explicit Init_WaterlinkedDvlFrame_transducer_nsd(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_transducer_beam_valid transducer_nsd(::tauv_msgs::msg::WaterlinkedDvlFrame::_transducer_nsd_type arg)
  {
    msg_.transducer_nsd = std::move(arg);
    return Init_WaterlinkedDvlFrame_transducer_beam_valid(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_transducer_rssi
{
public:
  explicit Init_WaterlinkedDvlFrame_transducer_rssi(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_transducer_nsd transducer_rssi(::tauv_msgs::msg::WaterlinkedDvlFrame::_transducer_rssi_type arg)
  {
    msg_.transducer_rssi = std::move(arg);
    return Init_WaterlinkedDvlFrame_transducer_nsd(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_transducer_distance
{
public:
  explicit Init_WaterlinkedDvlFrame_transducer_distance(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_transducer_rssi transducer_distance(::tauv_msgs::msg::WaterlinkedDvlFrame::_transducer_distance_type arg)
  {
    msg_.transducer_distance = std::move(arg);
    return Init_WaterlinkedDvlFrame_transducer_rssi(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_transducer_velocity
{
public:
  explicit Init_WaterlinkedDvlFrame_transducer_velocity(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_transducer_distance transducer_velocity(::tauv_msgs::msg::WaterlinkedDvlFrame::_transducer_velocity_type arg)
  {
    msg_.transducer_velocity = std::move(arg);
    return Init_WaterlinkedDvlFrame_transducer_distance(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_altitude
{
public:
  explicit Init_WaterlinkedDvlFrame_altitude(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_transducer_velocity altitude(::tauv_msgs::msg::WaterlinkedDvlFrame::_altitude_type arg)
  {
    msg_.altitude = std::move(arg);
    return Init_WaterlinkedDvlFrame_transducer_velocity(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_covariance
{
public:
  explicit Init_WaterlinkedDvlFrame_covariance(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_altitude covariance(::tauv_msgs::msg::WaterlinkedDvlFrame::_covariance_type arg)
  {
    msg_.covariance = std::move(arg);
    return Init_WaterlinkedDvlFrame_altitude(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_fom
{
public:
  explicit Init_WaterlinkedDvlFrame_fom(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_covariance fom(::tauv_msgs::msg::WaterlinkedDvlFrame::_fom_type arg)
  {
    msg_.fom = std::move(arg);
    return Init_WaterlinkedDvlFrame_covariance(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_vz
{
public:
  explicit Init_WaterlinkedDvlFrame_vz(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_fom vz(::tauv_msgs::msg::WaterlinkedDvlFrame::_vz_type arg)
  {
    msg_.vz = std::move(arg);
    return Init_WaterlinkedDvlFrame_fom(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_vy
{
public:
  explicit Init_WaterlinkedDvlFrame_vy(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_vz vy(::tauv_msgs::msg::WaterlinkedDvlFrame::_vy_type arg)
  {
    msg_.vy = std::move(arg);
    return Init_WaterlinkedDvlFrame_vz(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_vx
{
public:
  explicit Init_WaterlinkedDvlFrame_vx(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_vy vx(::tauv_msgs::msg::WaterlinkedDvlFrame::_vx_type arg)
  {
    msg_.vx = std::move(arg);
    return Init_WaterlinkedDvlFrame_vy(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_time
{
public:
  explicit Init_WaterlinkedDvlFrame_time(::tauv_msgs::msg::WaterlinkedDvlFrame & msg)
  : msg_(msg)
  {}
  Init_WaterlinkedDvlFrame_vx time(::tauv_msgs::msg::WaterlinkedDvlFrame::_time_type arg)
  {
    msg_.time = std::move(arg);
    return Init_WaterlinkedDvlFrame_vx(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

class Init_WaterlinkedDvlFrame_header
{
public:
  Init_WaterlinkedDvlFrame_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_WaterlinkedDvlFrame_time header(::tauv_msgs::msg::WaterlinkedDvlFrame::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_WaterlinkedDvlFrame_time(msg_);
  }

private:
  ::tauv_msgs::msg::WaterlinkedDvlFrame msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tauv_msgs::msg::WaterlinkedDvlFrame>()
{
  return tauv_msgs::msg::builder::Init_WaterlinkedDvlFrame_header();
}

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__BUILDER_HPP_
