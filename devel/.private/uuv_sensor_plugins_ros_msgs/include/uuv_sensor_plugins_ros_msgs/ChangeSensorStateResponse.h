// Generated by gencpp from file uuv_sensor_plugins_ros_msgs/ChangeSensorStateResponse.msg
// DO NOT EDIT!


#ifndef UUV_SENSOR_PLUGINS_ROS_MSGS_MESSAGE_CHANGESENSORSTATERESPONSE_H
#define UUV_SENSOR_PLUGINS_ROS_MSGS_MESSAGE_CHANGESENSORSTATERESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace uuv_sensor_plugins_ros_msgs
{
template <class ContainerAllocator>
struct ChangeSensorStateResponse_
{
  typedef ChangeSensorStateResponse_<ContainerAllocator> Type;

  ChangeSensorStateResponse_()
    : success(false)
    , message()  {
    }
  ChangeSensorStateResponse_(const ContainerAllocator& _alloc)
    : success(false)
    , message(_alloc)  {
  (void)_alloc;
    }



   typedef uint8_t _success_type;
  _success_type success;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _message_type;
  _message_type message;





  typedef boost::shared_ptr< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> const> ConstPtr;

}; // struct ChangeSensorStateResponse_

typedef ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<std::allocator<void> > ChangeSensorStateResponse;

typedef boost::shared_ptr< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse > ChangeSensorStateResponsePtr;
typedef boost::shared_ptr< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse const> ChangeSensorStateResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator1> & lhs, const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator2> & rhs)
{
  return lhs.success == rhs.success &&
    lhs.message == rhs.message;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator1> & lhs, const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace uuv_sensor_plugins_ros_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "937c9679a518e3a18d831e57125ea522";
  }

  static const char* value(const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x937c9679a518e3a1ULL;
  static const uint64_t static_value2 = 0x8d831e57125ea522ULL;
};

template<class ContainerAllocator>
struct DataType< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uuv_sensor_plugins_ros_msgs/ChangeSensorStateResponse";
  }

  static const char* value(const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "bool success\n"
"string message\n"
"\n"
;
  }

  static const char* value(const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.success);
      stream.next(m.message);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ChangeSensorStateResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::uuv_sensor_plugins_ros_msgs::ChangeSensorStateResponse_<ContainerAllocator>& v)
  {
    s << indent << "success: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.success);
    s << indent << "message: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.message);
  }
};

} // namespace message_operations
} // namespace ros

#endif // UUV_SENSOR_PLUGINS_ROS_MSGS_MESSAGE_CHANGESENSORSTATERESPONSE_H
