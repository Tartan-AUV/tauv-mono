// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tauv_msgs:msg/NavigationState.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/navigation_state.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__STRUCT_HPP_
#define TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__STRUCT_HPP_

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
// Member 'body_pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"
// Member 'v_b'
// Member 'a_b'
// Member 'omega_b'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tauv_msgs__msg__NavigationState __attribute__((deprecated))
#else
# define DEPRECATED__tauv_msgs__msg__NavigationState __declspec(deprecated)
#endif

namespace tauv_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct NavigationState_
{
  using Type = NavigationState_<ContainerAllocator>;

  explicit NavigationState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    body_pose(_init),
    v_b(_init),
    a_b(_init),
    omega_b(_init)
  {
    (void)_init;
  }

  explicit NavigationState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    body_pose(_alloc, _init),
    v_b(_alloc, _init),
    a_b(_alloc, _init),
    omega_b(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _body_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _body_pose_type body_pose;
  using _v_b_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _v_b_type v_b;
  using _a_b_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _a_b_type a_b;
  using _omega_b_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _omega_b_type omega_b;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__body_pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->body_pose = _arg;
    return *this;
  }
  Type & set__v_b(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->v_b = _arg;
    return *this;
  }
  Type & set__a_b(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->a_b = _arg;
    return *this;
  }
  Type & set__omega_b(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->omega_b = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tauv_msgs::msg::NavigationState_<ContainerAllocator> *;
  using ConstRawPtr =
    const tauv_msgs::msg::NavigationState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::NavigationState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::NavigationState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tauv_msgs__msg__NavigationState
    std::shared_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tauv_msgs__msg__NavigationState
    std::shared_ptr<tauv_msgs::msg::NavigationState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NavigationState_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->body_pose != other.body_pose) {
      return false;
    }
    if (this->v_b != other.v_b) {
      return false;
    }
    if (this->a_b != other.a_b) {
      return false;
    }
    if (this->omega_b != other.omega_b) {
      return false;
    }
    return true;
  }
  bool operator!=(const NavigationState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NavigationState_

// alias to use template instance with default allocator
using NavigationState =
  tauv_msgs::msg::NavigationState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__NAVIGATION_STATE__STRUCT_HPP_
