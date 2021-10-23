// Generated by gencpp from file tauv_common/RegisterMeasurementRequest.msg
// DO NOT EDIT!


#ifndef TAUV_COMMON_MESSAGE_REGISTERMEASUREMENTREQUEST_H
#define TAUV_COMMON_MESSAGE_REGISTERMEASUREMENTREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <tauv_msgs/PoseGraphMeasurement.h>

namespace tauv_common
{
template <class ContainerAllocator>
struct RegisterMeasurementRequest_
{
  typedef RegisterMeasurementRequest_<ContainerAllocator> Type;

  RegisterMeasurementRequest_()
    : pg_meas()  {
    }
  RegisterMeasurementRequest_(const ContainerAllocator& _alloc)
    : pg_meas(_alloc)  {
  (void)_alloc;
    }



   typedef  ::tauv_msgs::PoseGraphMeasurement_<ContainerAllocator>  _pg_meas_type;
  _pg_meas_type pg_meas;





  typedef boost::shared_ptr< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> const> ConstPtr;

}; // struct RegisterMeasurementRequest_

typedef ::tauv_common::RegisterMeasurementRequest_<std::allocator<void> > RegisterMeasurementRequest;

typedef boost::shared_ptr< ::tauv_common::RegisterMeasurementRequest > RegisterMeasurementRequestPtr;
typedef boost::shared_ptr< ::tauv_common::RegisterMeasurementRequest const> RegisterMeasurementRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator1> & lhs, const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator2> & rhs)
{
  return lhs.pg_meas == rhs.pg_meas;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator1> & lhs, const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace tauv_common

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "b355dd17bfdad2a0499de8384660e7ff";
  }

  static const char* value(const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xb355dd17bfdad2a0ULL;
  static const uint64_t static_value2 = 0x499de8384660e7ffULL;
};

template<class ContainerAllocator>
struct DataType< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tauv_common/RegisterMeasurementRequest";
  }

  static const char* value(const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tauv_msgs/PoseGraphMeasurement pg_meas\n"
"\n"
"================================================================================\n"
"MSG: tauv_msgs/PoseGraphMeasurement\n"
"Header header\n"
"uint32 landmark_id\n"
"geometry_msgs/Point position\n"
"\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
;
  }

  static const char* value(const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.pg_meas);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct RegisterMeasurementRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::tauv_common::RegisterMeasurementRequest_<ContainerAllocator>& v)
  {
    s << indent << "pg_meas: ";
    s << std::endl;
    Printer< ::tauv_msgs::PoseGraphMeasurement_<ContainerAllocator> >::stream(s, indent + "  ", v.pg_meas);
  }
};

} // namespace message_operations
} // namespace ros

#endif // TAUV_COMMON_MESSAGE_REGISTERMEASUREMENTREQUEST_H
