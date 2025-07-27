// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tauv_msgs:msg/TargetThrust.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/target_thrust.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__STRUCT_HPP_
#define TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__tauv_msgs__msg__TargetThrust __attribute__((deprecated))
#else
# define DEPRECATED__tauv_msgs__msg__TargetThrust __declspec(deprecated)
#endif

namespace tauv_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TargetThrust_
{
  using Type = TargetThrust_<ContainerAllocator>;

  explicit TargetThrust_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 8>::iterator, double>(this->target_thrust.begin(), this->target_thrust.end(), 0.0);
    }
  }

  explicit TargetThrust_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : target_thrust(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 8>::iterator, double>(this->target_thrust.begin(), this->target_thrust.end(), 0.0);
    }
  }

  // field types and members
  using _target_thrust_type =
    std::array<double, 8>;
  _target_thrust_type target_thrust;

  // setters for named parameter idiom
  Type & set__target_thrust(
    const std::array<double, 8> & _arg)
  {
    this->target_thrust = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tauv_msgs::msg::TargetThrust_<ContainerAllocator> *;
  using ConstRawPtr =
    const tauv_msgs::msg::TargetThrust_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::TargetThrust_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::TargetThrust_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tauv_msgs__msg__TargetThrust
    std::shared_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tauv_msgs__msg__TargetThrust
    std::shared_ptr<tauv_msgs::msg::TargetThrust_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TargetThrust_ & other) const
  {
    if (this->target_thrust != other.target_thrust) {
      return false;
    }
    return true;
  }
  bool operator!=(const TargetThrust_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TargetThrust_

// alias to use template instance with default allocator
using TargetThrust =
  tauv_msgs::msg::TargetThrust_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__TARGET_THRUST__STRUCT_HPP_
