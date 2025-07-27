// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tauv_msgs:msg/EscTelemetry.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/esc_telemetry.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__STRUCT_HPP_
#define TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__STRUCT_HPP_

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
# define DEPRECATED__tauv_msgs__msg__EscTelemetry __attribute__((deprecated))
#else
# define DEPRECATED__tauv_msgs__msg__EscTelemetry __declspec(deprecated)
#endif

namespace tauv_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct EscTelemetry_
{
  using Type = EscTelemetry_<ContainerAllocator>;

  explicit EscTelemetry_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->rpm = 0l;
      this->voltage = 0.0f;
      this->current = 0.0f;
      this->temperature = 0.0f;
      this->fault_code = 0;
    }
  }

  explicit EscTelemetry_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->rpm = 0l;
      this->voltage = 0.0f;
      this->current = 0.0f;
      this->temperature = 0.0f;
      this->fault_code = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _id_type =
    uint8_t;
  _id_type id;
  using _rpm_type =
    int32_t;
  _rpm_type rpm;
  using _voltage_type =
    float;
  _voltage_type voltage;
  using _current_type =
    float;
  _current_type current;
  using _temperature_type =
    float;
  _temperature_type temperature;
  using _fault_code_type =
    uint8_t;
  _fault_code_type fault_code;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__id(
    const uint8_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__rpm(
    const int32_t & _arg)
  {
    this->rpm = _arg;
    return *this;
  }
  Type & set__voltage(
    const float & _arg)
  {
    this->voltage = _arg;
    return *this;
  }
  Type & set__current(
    const float & _arg)
  {
    this->current = _arg;
    return *this;
  }
  Type & set__temperature(
    const float & _arg)
  {
    this->temperature = _arg;
    return *this;
  }
  Type & set__fault_code(
    const uint8_t & _arg)
  {
    this->fault_code = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tauv_msgs::msg::EscTelemetry_<ContainerAllocator> *;
  using ConstRawPtr =
    const tauv_msgs::msg::EscTelemetry_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::EscTelemetry_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::EscTelemetry_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tauv_msgs__msg__EscTelemetry
    std::shared_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tauv_msgs__msg__EscTelemetry
    std::shared_ptr<tauv_msgs::msg::EscTelemetry_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EscTelemetry_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->id != other.id) {
      return false;
    }
    if (this->rpm != other.rpm) {
      return false;
    }
    if (this->voltage != other.voltage) {
      return false;
    }
    if (this->current != other.current) {
      return false;
    }
    if (this->temperature != other.temperature) {
      return false;
    }
    if (this->fault_code != other.fault_code) {
      return false;
    }
    return true;
  }
  bool operator!=(const EscTelemetry_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EscTelemetry_

// alias to use template instance with default allocator
using EscTelemetry =
  tauv_msgs::msg::EscTelemetry_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__ESC_TELEMETRY__STRUCT_HPP_
