// Generated by gencpp from file tauv_msgs/TunePidResponse.msg
// DO NOT EDIT!


#ifndef TAUV_MSGS_MESSAGE_TUNEPIDRESPONSE_H
#define TAUV_MSGS_MESSAGE_TUNEPIDRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace tauv_msgs
{
template <class ContainerAllocator>
struct TunePidResponse_
{
  typedef TunePidResponse_<ContainerAllocator> Type;

  TunePidResponse_()
    : success(false)  {
    }
  TunePidResponse_(const ContainerAllocator& _alloc)
    : success(false)  {
  (void)_alloc;
    }



   typedef uint8_t _success_type;
  _success_type success;





  typedef boost::shared_ptr< ::tauv_msgs::TunePidResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::tauv_msgs::TunePidResponse_<ContainerAllocator> const> ConstPtr;

}; // struct TunePidResponse_

typedef ::tauv_msgs::TunePidResponse_<std::allocator<void> > TunePidResponse;

typedef boost::shared_ptr< ::tauv_msgs::TunePidResponse > TunePidResponsePtr;
typedef boost::shared_ptr< ::tauv_msgs::TunePidResponse const> TunePidResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::tauv_msgs::TunePidResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::tauv_msgs::TunePidResponse_<ContainerAllocator1> & lhs, const ::tauv_msgs::TunePidResponse_<ContainerAllocator2> & rhs)
{
  return lhs.success == rhs.success;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::tauv_msgs::TunePidResponse_<ContainerAllocator1> & lhs, const ::tauv_msgs::TunePidResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace tauv_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tauv_msgs::TunePidResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tauv_msgs::TunePidResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tauv_msgs::TunePidResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "358e233cde0c8a8bcfea4ce193f8fc15";
  }

  static const char* value(const ::tauv_msgs::TunePidResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x358e233cde0c8a8bULL;
  static const uint64_t static_value2 = 0xcfea4ce193f8fc15ULL;
};

template<class ContainerAllocator>
struct DataType< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tauv_msgs/TunePidResponse";
  }

  static const char* value(const ::tauv_msgs::TunePidResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "bool success\n"
;
  }

  static const char* value(const ::tauv_msgs::TunePidResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.success);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct TunePidResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::tauv_msgs::TunePidResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::tauv_msgs::TunePidResponse_<ContainerAllocator>& v)
  {
    s << indent << "success: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.success);
  }
};

} // namespace message_operations
} // namespace ros

#endif // TAUV_MSGS_MESSAGE_TUNEPIDRESPONSE_H
