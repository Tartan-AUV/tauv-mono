// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/waterlinked_dvl_frame.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__STRUCT_HPP_
#define TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__STRUCT_HPP_

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
# define DEPRECATED__tauv_msgs__msg__WaterlinkedDvlFrame __attribute__((deprecated))
#else
# define DEPRECATED__tauv_msgs__msg__WaterlinkedDvlFrame __declspec(deprecated)
#endif

namespace tauv_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct WaterlinkedDvlFrame_
{
  using Type = WaterlinkedDvlFrame_<ContainerAllocator>;

  explicit WaterlinkedDvlFrame_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time = 0.0;
      this->vx = 0.0;
      this->vy = 0.0;
      this->vz = 0.0;
      this->fom = 0.0;
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
      this->altitude = 0.0;
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_velocity.begin(), this->transducer_velocity.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_distance.begin(), this->transducer_distance.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_rssi.begin(), this->transducer_rssi.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_nsd.begin(), this->transducer_nsd.end(), 0.0);
      std::fill<typename std::array<bool, 4>::iterator, bool>(this->transducer_beam_valid.begin(), this->transducer_beam_valid.end(), false);
      this->velocity_valid = false;
      this->status = 0l;
      this->time_of_validity = 0ll;
      this->time_of_transmission = 0ll;
    }
  }

  explicit WaterlinkedDvlFrame_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    covariance(_alloc),
    transducer_velocity(_alloc),
    transducer_distance(_alloc),
    transducer_rssi(_alloc),
    transducer_nsd(_alloc),
    transducer_beam_valid(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->time = 0.0;
      this->vx = 0.0;
      this->vy = 0.0;
      this->vz = 0.0;
      this->fom = 0.0;
      std::fill<typename std::array<double, 9>::iterator, double>(this->covariance.begin(), this->covariance.end(), 0.0);
      this->altitude = 0.0;
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_velocity.begin(), this->transducer_velocity.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_distance.begin(), this->transducer_distance.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_rssi.begin(), this->transducer_rssi.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->transducer_nsd.begin(), this->transducer_nsd.end(), 0.0);
      std::fill<typename std::array<bool, 4>::iterator, bool>(this->transducer_beam_valid.begin(), this->transducer_beam_valid.end(), false);
      this->velocity_valid = false;
      this->status = 0l;
      this->time_of_validity = 0ll;
      this->time_of_transmission = 0ll;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _time_type =
    double;
  _time_type time;
  using _vx_type =
    double;
  _vx_type vx;
  using _vy_type =
    double;
  _vy_type vy;
  using _vz_type =
    double;
  _vz_type vz;
  using _fom_type =
    double;
  _fom_type fom;
  using _covariance_type =
    std::array<double, 9>;
  _covariance_type covariance;
  using _altitude_type =
    double;
  _altitude_type altitude;
  using _transducer_velocity_type =
    std::array<double, 4>;
  _transducer_velocity_type transducer_velocity;
  using _transducer_distance_type =
    std::array<double, 4>;
  _transducer_distance_type transducer_distance;
  using _transducer_rssi_type =
    std::array<double, 4>;
  _transducer_rssi_type transducer_rssi;
  using _transducer_nsd_type =
    std::array<double, 4>;
  _transducer_nsd_type transducer_nsd;
  using _transducer_beam_valid_type =
    std::array<bool, 4>;
  _transducer_beam_valid_type transducer_beam_valid;
  using _velocity_valid_type =
    bool;
  _velocity_valid_type velocity_valid;
  using _status_type =
    int32_t;
  _status_type status;
  using _time_of_validity_type =
    int64_t;
  _time_of_validity_type time_of_validity;
  using _time_of_transmission_type =
    int64_t;
  _time_of_transmission_type time_of_transmission;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__time(
    const double & _arg)
  {
    this->time = _arg;
    return *this;
  }
  Type & set__vx(
    const double & _arg)
  {
    this->vx = _arg;
    return *this;
  }
  Type & set__vy(
    const double & _arg)
  {
    this->vy = _arg;
    return *this;
  }
  Type & set__vz(
    const double & _arg)
  {
    this->vz = _arg;
    return *this;
  }
  Type & set__fom(
    const double & _arg)
  {
    this->fom = _arg;
    return *this;
  }
  Type & set__covariance(
    const std::array<double, 9> & _arg)
  {
    this->covariance = _arg;
    return *this;
  }
  Type & set__altitude(
    const double & _arg)
  {
    this->altitude = _arg;
    return *this;
  }
  Type & set__transducer_velocity(
    const std::array<double, 4> & _arg)
  {
    this->transducer_velocity = _arg;
    return *this;
  }
  Type & set__transducer_distance(
    const std::array<double, 4> & _arg)
  {
    this->transducer_distance = _arg;
    return *this;
  }
  Type & set__transducer_rssi(
    const std::array<double, 4> & _arg)
  {
    this->transducer_rssi = _arg;
    return *this;
  }
  Type & set__transducer_nsd(
    const std::array<double, 4> & _arg)
  {
    this->transducer_nsd = _arg;
    return *this;
  }
  Type & set__transducer_beam_valid(
    const std::array<bool, 4> & _arg)
  {
    this->transducer_beam_valid = _arg;
    return *this;
  }
  Type & set__velocity_valid(
    const bool & _arg)
  {
    this->velocity_valid = _arg;
    return *this;
  }
  Type & set__status(
    const int32_t & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__time_of_validity(
    const int64_t & _arg)
  {
    this->time_of_validity = _arg;
    return *this;
  }
  Type & set__time_of_transmission(
    const int64_t & _arg)
  {
    this->time_of_transmission = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator> *;
  using ConstRawPtr =
    const tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tauv_msgs__msg__WaterlinkedDvlFrame
    std::shared_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tauv_msgs__msg__WaterlinkedDvlFrame
    std::shared_ptr<tauv_msgs::msg::WaterlinkedDvlFrame_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const WaterlinkedDvlFrame_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->time != other.time) {
      return false;
    }
    if (this->vx != other.vx) {
      return false;
    }
    if (this->vy != other.vy) {
      return false;
    }
    if (this->vz != other.vz) {
      return false;
    }
    if (this->fom != other.fom) {
      return false;
    }
    if (this->covariance != other.covariance) {
      return false;
    }
    if (this->altitude != other.altitude) {
      return false;
    }
    if (this->transducer_velocity != other.transducer_velocity) {
      return false;
    }
    if (this->transducer_distance != other.transducer_distance) {
      return false;
    }
    if (this->transducer_rssi != other.transducer_rssi) {
      return false;
    }
    if (this->transducer_nsd != other.transducer_nsd) {
      return false;
    }
    if (this->transducer_beam_valid != other.transducer_beam_valid) {
      return false;
    }
    if (this->velocity_valid != other.velocity_valid) {
      return false;
    }
    if (this->status != other.status) {
      return false;
    }
    if (this->time_of_validity != other.time_of_validity) {
      return false;
    }
    if (this->time_of_transmission != other.time_of_transmission) {
      return false;
    }
    return true;
  }
  bool operator!=(const WaterlinkedDvlFrame_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct WaterlinkedDvlFrame_

// alias to use template instance with default allocator
using WaterlinkedDvlFrame =
  tauv_msgs::msg::WaterlinkedDvlFrame_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tauv_msgs

#endif  // TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__STRUCT_HPP_
