// Generated by gencpp from file tauv_common/ThrusterManagerInfoRequest.msg
// DO NOT EDIT!


#ifndef TAUV_COMMON_MESSAGE_THRUSTERMANAGERINFOREQUEST_H
#define TAUV_COMMON_MESSAGE_THRUSTERMANAGERINFOREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace tauv_common
{
template <class ContainerAllocator>
struct ThrusterManagerInfoRequest_
{
  typedef ThrusterManagerInfoRequest_<ContainerAllocator> Type;

  ThrusterManagerInfoRequest_()
    {
    }
  ThrusterManagerInfoRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> const> ConstPtr;

}; // struct ThrusterManagerInfoRequest_

typedef ::tauv_common::ThrusterManagerInfoRequest_<std::allocator<void> > ThrusterManagerInfoRequest;

typedef boost::shared_ptr< ::tauv_common::ThrusterManagerInfoRequest > ThrusterManagerInfoRequestPtr;
typedef boost::shared_ptr< ::tauv_common::ThrusterManagerInfoRequest const> ThrusterManagerInfoRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


} // namespace tauv_common

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tauv_common/ThrusterManagerInfoRequest";
  }

  static const char* value(const ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Copyright (c) 2016 The UUV Simulator Authors.\n"
"# All rights reserved.\n"
"#\n"
"# Licensed under the Apache License, Version 2.0 (the \"License\");\n"
"# you may not use this file except in compliance with the License.\n"
"# You may obtain a copy of the License at\n"
"#\n"
"#     http://www.apache.org/licenses/LICENSE-2.0\n"
"#\n"
"# Unless required by applicable law or agreed to in writing, software\n"
"# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
"# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
"# See the License for the specific language governing permissions and\n"
"# limitations under the License.\n"
"\n"
;
  }

  static const char* value(const ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ThrusterManagerInfoRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::tauv_common::ThrusterManagerInfoRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // TAUV_COMMON_MESSAGE_THRUSTERMANAGERINFOREQUEST_H
