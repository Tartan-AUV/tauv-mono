// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tauv_msgs:msg/RpmCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/rpm_command.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__STRUCT_HPP_
#define TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__tauv_msgs__msg__RpmCommand __attribute__((deprecated))
#else
# define DEPRECATED__tauv_msgs__msg__RpmCommand __declspec(deprecated)
#endif

namespace tauv_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RpmCommand_
{
  using Type = RpmCommand_<ContainerAllocator>;

  explicit RpmCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<int32_t, 8>::iterator, int32_t>(this->rpms.begin(), this->rpms.end(), 0l);
      std::fill<typename std::array<uint8_t, 8>::iterator, uint8_t>(this->enables.begin(), this->enables.end(), 0);
    }
  }

  explicit RpmCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : rpms(_alloc),
    enables(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<int32_t, 8>::iterator, int32_t>(this->rpms.begin(), this->rpms.end(), 0l);
      std::fill<typename std::array<uint8_t, 8>::iterator, uint8_t>(this->enables.begin(), this->enables.end(), 0);
    }
  }

  // field types and members
  using _rpms_type =
    std::array<int32_t, 8>;
  _rpms_type rpms;
  using _enables_type =
    std::array<uint8_t, 8>;
  _enables_type enables;

  // setters for named parameter idiom
  Type & set__rpms(
    const std::array<int32_t, 8> & _arg)
  {
    this->rpms = _arg;
    return *this;
  }
  Type & set__enables(
    const std::array<uint8_t, 8> & _arg)
  {
    this->enables = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tauv_msgs::msg::RpmCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const tauv_msgs::msg::RpmCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::RpmCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::RpmCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tauv_msgs__msg__RpmCommand
    std::shared_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tauv_msgs__msg__RpmCommand
    std::shared_ptr<tauv_msgs::msg::RpmCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RpmCommand_ & other) const
  {
    if (this->rpms != other.rpms) {
      return false;
    }
    if (this->enables != other.enables) {
      return false;
    }
    return true;
  }
  bool operator!=(const RpmCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RpmCommand_

// alias to use template instance with default allocator
using RpmCommand =
  tauv_msgs::msg::RpmCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__RPM_COMMAND__STRUCT_HPP_
