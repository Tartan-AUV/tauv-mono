// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tauv_msgs:msg/VelocityAttitudeCommand.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/velocity_attitude_command.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__STRUCT_HPP_
#define TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__STRUCT_HPP_

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
// Member 'target_velocity'
// Member 'feedforward_acceleration'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"
// Member 'target_attitude'
#include "geometry_msgs/msg/detail/quaternion__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tauv_msgs__msg__VelocityAttitudeCommand __attribute__((deprecated))
#else
# define DEPRECATED__tauv_msgs__msg__VelocityAttitudeCommand __declspec(deprecated)
#endif

namespace tauv_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct VelocityAttitudeCommand_
{
  using Type = VelocityAttitudeCommand_<ContainerAllocator>;

  explicit VelocityAttitudeCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    target_velocity(_init),
    target_attitude(_init),
    feedforward_acceleration(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->velocity_control_enabled = false;
      this->attitude_control_enabled = false;
    }
  }

  explicit VelocityAttitudeCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    target_velocity(_alloc, _init),
    target_attitude(_alloc, _init),
    feedforward_acceleration(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->velocity_control_enabled = false;
      this->attitude_control_enabled = false;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _target_velocity_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _target_velocity_type target_velocity;
  using _target_attitude_type =
    geometry_msgs::msg::Quaternion_<ContainerAllocator>;
  _target_attitude_type target_attitude;
  using _feedforward_acceleration_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _feedforward_acceleration_type feedforward_acceleration;
  using _velocity_control_enabled_type =
    bool;
  _velocity_control_enabled_type velocity_control_enabled;
  using _attitude_control_enabled_type =
    bool;
  _attitude_control_enabled_type attitude_control_enabled;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__target_velocity(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->target_velocity = _arg;
    return *this;
  }
  Type & set__target_attitude(
    const geometry_msgs::msg::Quaternion_<ContainerAllocator> & _arg)
  {
    this->target_attitude = _arg;
    return *this;
  }
  Type & set__feedforward_acceleration(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->feedforward_acceleration = _arg;
    return *this;
  }
  Type & set__velocity_control_enabled(
    const bool & _arg)
  {
    this->velocity_control_enabled = _arg;
    return *this;
  }
  Type & set__attitude_control_enabled(
    const bool & _arg)
  {
    this->attitude_control_enabled = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tauv_msgs__msg__VelocityAttitudeCommand
    std::shared_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tauv_msgs__msg__VelocityAttitudeCommand
    std::shared_ptr<tauv_msgs::msg::VelocityAttitudeCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const VelocityAttitudeCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->target_velocity != other.target_velocity) {
      return false;
    }
    if (this->target_attitude != other.target_attitude) {
      return false;
    }
    if (this->feedforward_acceleration != other.feedforward_acceleration) {
      return false;
    }
    if (this->velocity_control_enabled != other.velocity_control_enabled) {
      return false;
    }
    if (this->attitude_control_enabled != other.attitude_control_enabled) {
      return false;
    }
    return true;
  }
  bool operator!=(const VelocityAttitudeCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct VelocityAttitudeCommand_

// alias to use template instance with default allocator
using VelocityAttitudeCommand =
  tauv_msgs::msg::VelocityAttitudeCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__VELOCITY_ATTITUDE_COMMAND__STRUCT_HPP_
