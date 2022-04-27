#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include "transforms.h"

namespace transforms {
  TransformClient::TransformClient(ros::NodeHandle &n) : tf_listener(n) 
  {
    
  }

  bool TransformClient::transform_point(Eigen::Vector3d &v, Eigen::Vector3d &r, Frame from, Frame to)
  {
    tf::StampedTransform tsf; 

    try {
      this->tf_listener.lookupTransform(this->frame_ids.at(to), this->frame_ids.at(from), ros::Time(0.0), tsf);
    } catch (tf::TransformException &ex) {
      ROS_ERROR("%s", ex.what());
      return false;
    }

    return true;
  }
}
