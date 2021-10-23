// Generated by gencpp from file uuv_gazebo_ros_plugins_msgs/SetUseGlobalCurrentVelRequest.msg
// DO NOT EDIT!


#ifndef UUV_GAZEBO_ROS_PLUGINS_MSGS_MESSAGE_SETUSEGLOBALCURRENTVELREQUEST_H
#define UUV_GAZEBO_ROS_PLUGINS_MSGS_MESSAGE_SETUSEGLOBALCURRENTVELREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace uuv_gazebo_ros_plugins_msgs
{
template <class ContainerAllocator>
struct SetUseGlobalCurrentVelRequest_
{
  typedef SetUseGlobalCurrentVelRequest_<ContainerAllocator> Type;

  SetUseGlobalCurrentVelRequest_()
    : use_global(false)  {
    }
  SetUseGlobalCurrentVelRequest_(const ContainerAllocator& _alloc)
    : use_global(false)  {
  (void)_alloc;
    }



   typedef uint8_t _use_global_type;
  _use_global_type use_global;





  typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> const> ConstPtr;

}; // struct SetUseGlobalCurrentVelRequest_

typedef ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<std::allocator<void> > SetUseGlobalCurrentVelRequest;

typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest > SetUseGlobalCurrentVelRequestPtr;
typedef boost::shared_ptr< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest const> SetUseGlobalCurrentVelRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator1> & lhs, const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator2> & rhs)
{
  return lhs.use_global == rhs.use_global;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator1> & lhs, const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace uuv_gazebo_ros_plugins_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "cb3581ad5adb4e1f612596312cf9e4fe";
  }

  static const char* value(const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xcb3581ad5adb4e1fULL;
  static const uint64_t static_value2 = 0x612596312cf9e4feULL;
};

template<class ContainerAllocator>
struct DataType< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uuv_gazebo_ros_plugins_msgs/SetUseGlobalCurrentVelRequest";
  }

  static const char* value(const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
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
"bool use_global\n"
;
  }

  static const char* value(const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.use_global);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SetUseGlobalCurrentVelRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::uuv_gazebo_ros_plugins_msgs::SetUseGlobalCurrentVelRequest_<ContainerAllocator>& v)
  {
    s << indent << "use_global: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.use_global);
  }
};

} // namespace message_operations
} // namespace ros

#endif // UUV_GAZEBO_ROS_PLUGINS_MSGS_MESSAGE_SETUSEGLOBALCURRENTVELREQUEST_H
