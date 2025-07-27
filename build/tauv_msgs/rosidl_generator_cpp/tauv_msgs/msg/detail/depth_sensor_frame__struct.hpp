// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tauv_msgs:msg/DepthSensorFrame.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/depth_sensor_frame.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__STRUCT_HPP_
#define TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tauv_msgs__msg__DepthSensorFrame __attribute__((deprecated))
#else
# define DEPRECATED__tauv_msgs__msg__DepthSensorFrame __declspec(deprecated)
#endif

namespace tauv_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DepthSensorFrame_
{
  using Type = DepthSensorFrame_<ContainerAllocator>;

  explicit DepthSensorFrame_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->depth = 0.0f;
      this->pressure = 0.0f;
      this->temperature = 0.0f;
    }
  }

  explicit DepthSensorFrame_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->depth = 0.0f;
      this->pressure = 0.0f;
      this->temperature = 0.0f;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _depth_type =
    float;
  _depth_type depth;
  using _pressure_type =
    float;
  _pressure_type pressure;
  using _temperature_type =
    float;
  _temperature_type temperature;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__depth(
    const float & _arg)
  {
    this->depth = _arg;
    return *this;
  }
  Type & set__pressure(
    const float & _arg)
  {
    this->pressure = _arg;
    return *this;
  }
  Type & set__temperature(
    const float & _arg)
  {
    this->temperature = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator> *;
  using ConstRawPtr =
    const tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tauv_msgs__msg__DepthSensorFrame
    std::shared_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tauv_msgs__msg__DepthSensorFrame
    std::shared_ptr<tauv_msgs::msg::DepthSensorFrame_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DepthSensorFrame_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->depth != other.depth) {
      return false;
    }
    if (this->pressure != other.pressure) {
      return false;
    }
    if (this->temperature != other.temperature) {
      return false;
    }
    return true;
  }
  bool operator!=(const DepthSensorFrame_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DepthSensorFrame_

// alias to use template instance with default allocator
using DepthSensorFrame =
  tauv_msgs::msg::DepthSensorFrame_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__DEPTH_SENSOR_FRAME__STRUCT_HPP_
