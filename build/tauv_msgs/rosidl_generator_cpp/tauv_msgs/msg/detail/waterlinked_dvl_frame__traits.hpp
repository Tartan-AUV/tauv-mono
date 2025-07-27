// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tauv_msgs:msg/WaterlinkedDvlFrame.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "tauv_msgs/msg/waterlinked_dvl_frame.hpp"


#ifndef TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__TRAITS_HPP_
#define TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tauv_msgs/msg/detail/waterlinked_dvl_frame__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace tauv_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const WaterlinkedDvlFrame & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: time
  {
    out << "time: ";
    rosidl_generator_traits::value_to_yaml(msg.time, out);
    out << ", ";
  }

  // member: vx
  {
    out << "vx: ";
    rosidl_generator_traits::value_to_yaml(msg.vx, out);
    out << ", ";
  }

  // member: vy
  {
    out << "vy: ";
    rosidl_generator_traits::value_to_yaml(msg.vy, out);
    out << ", ";
  }

  // member: vz
  {
    out << "vz: ";
    rosidl_generator_traits::value_to_yaml(msg.vz, out);
    out << ", ";
  }

  // member: fom
  {
    out << "fom: ";
    rosidl_generator_traits::value_to_yaml(msg.fom, out);
    out << ", ";
  }

  // member: covariance
  {
    if (msg.covariance.size() == 0) {
      out << "covariance: []";
    } else {
      out << "covariance: [";
      size_t pending_items = msg.covariance.size();
      for (auto item : msg.covariance) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: altitude
  {
    out << "altitude: ";
    rosidl_generator_traits::value_to_yaml(msg.altitude, out);
    out << ", ";
  }

  // member: transducer_velocity
  {
    if (msg.transducer_velocity.size() == 0) {
      out << "transducer_velocity: []";
    } else {
      out << "transducer_velocity: [";
      size_t pending_items = msg.transducer_velocity.size();
      for (auto item : msg.transducer_velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: transducer_distance
  {
    if (msg.transducer_distance.size() == 0) {
      out << "transducer_distance: []";
    } else {
      out << "transducer_distance: [";
      size_t pending_items = msg.transducer_distance.size();
      for (auto item : msg.transducer_distance) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: transducer_rssi
  {
    if (msg.transducer_rssi.size() == 0) {
      out << "transducer_rssi: []";
    } else {
      out << "transducer_rssi: [";
      size_t pending_items = msg.transducer_rssi.size();
      for (auto item : msg.transducer_rssi) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: transducer_nsd
  {
    if (msg.transducer_nsd.size() == 0) {
      out << "transducer_nsd: []";
    } else {
      out << "transducer_nsd: [";
      size_t pending_items = msg.transducer_nsd.size();
      for (auto item : msg.transducer_nsd) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: transducer_beam_valid
  {
    if (msg.transducer_beam_valid.size() == 0) {
      out << "transducer_beam_valid: []";
    } else {
      out << "transducer_beam_valid: [";
      size_t pending_items = msg.transducer_beam_valid.size();
      for (auto item : msg.transducer_beam_valid) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: velocity_valid
  {
    out << "velocity_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity_valid, out);
    out << ", ";
  }

  // member: status
  {
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << ", ";
  }

  // member: time_of_validity
  {
    out << "time_of_validity: ";
    rosidl_generator_traits::value_to_yaml(msg.time_of_validity, out);
    out << ", ";
  }

  // member: time_of_transmission
  {
    out << "time_of_transmission: ";
    rosidl_generator_traits::value_to_yaml(msg.time_of_transmission, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const WaterlinkedDvlFrame & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time: ";
    rosidl_generator_traits::value_to_yaml(msg.time, out);
    out << "\n";
  }

  // member: vx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vx: ";
    rosidl_generator_traits::value_to_yaml(msg.vx, out);
    out << "\n";
  }

  // member: vy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vy: ";
    rosidl_generator_traits::value_to_yaml(msg.vy, out);
    out << "\n";
  }

  // member: vz
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vz: ";
    rosidl_generator_traits::value_to_yaml(msg.vz, out);
    out << "\n";
  }

  // member: fom
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "fom: ";
    rosidl_generator_traits::value_to_yaml(msg.fom, out);
    out << "\n";
  }

  // member: covariance
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.covariance.size() == 0) {
      out << "covariance: []\n";
    } else {
      out << "covariance:\n";
      for (auto item : msg.covariance) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: altitude
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "altitude: ";
    rosidl_generator_traits::value_to_yaml(msg.altitude, out);
    out << "\n";
  }

  // member: transducer_velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.transducer_velocity.size() == 0) {
      out << "transducer_velocity: []\n";
    } else {
      out << "transducer_velocity:\n";
      for (auto item : msg.transducer_velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: transducer_distance
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.transducer_distance.size() == 0) {
      out << "transducer_distance: []\n";
    } else {
      out << "transducer_distance:\n";
      for (auto item : msg.transducer_distance) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: transducer_rssi
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.transducer_rssi.size() == 0) {
      out << "transducer_rssi: []\n";
    } else {
      out << "transducer_rssi:\n";
      for (auto item : msg.transducer_rssi) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: transducer_nsd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.transducer_nsd.size() == 0) {
      out << "transducer_nsd: []\n";
    } else {
      out << "transducer_nsd:\n";
      for (auto item : msg.transducer_nsd) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: transducer_beam_valid
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.transducer_beam_valid.size() == 0) {
      out << "transducer_beam_valid: []\n";
    } else {
      out << "transducer_beam_valid:\n";
      for (auto item : msg.transducer_beam_valid) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: velocity_valid
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "velocity_valid: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity_valid, out);
    out << "\n";
  }

  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    rosidl_generator_traits::value_to_yaml(msg.status, out);
    out << "\n";
  }

  // member: time_of_validity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time_of_validity: ";
    rosidl_generator_traits::value_to_yaml(msg.time_of_validity, out);
    out << "\n";
  }

  // member: time_of_transmission
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "time_of_transmission: ";
    rosidl_generator_traits::value_to_yaml(msg.time_of_transmission, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const WaterlinkedDvlFrame & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace tauv_msgs

namespace rosidl_generator_traits
{

[[deprecated("use tauv_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tauv_msgs::msg::WaterlinkedDvlFrame & msg,
  std::ostream & out, size_t indentation = 0)
{
  tauv_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tauv_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const tauv_msgs::msg::WaterlinkedDvlFrame & msg)
{
  return tauv_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tauv_msgs::msg::WaterlinkedDvlFrame>()
{
  return "tauv_msgs::msg::WaterlinkedDvlFrame";
}

template<>
inline const char * name<tauv_msgs::msg::WaterlinkedDvlFrame>()
{
  return "tauv_msgs/msg/WaterlinkedDvlFrame";
}

template<>
struct has_fixed_size<tauv_msgs::msg::WaterlinkedDvlFrame>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<tauv_msgs::msg::WaterlinkedDvlFrame>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<tauv_msgs::msg::WaterlinkedDvlFrame>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TAUV_MSGS__MSG__DETAIL__WATERLINKED_DVL_FRAME__TRAITS_HPP_
