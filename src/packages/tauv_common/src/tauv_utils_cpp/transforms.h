#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tf/transform_listener.h>

#pragma once

namespace transforms {
  enum Frame {
    WORLD,
    ODOM,
    VEHICLE,
    IMU,
    DVL,
  };

  class TransformClient {
    public:
      TransformClient(ros::NodeHandle &n);

      bool transform_point(Eigen::Vector3d &v, Eigen::Vector3d &r, Frame from, Frame to);
      bool transform_vector(Eigen::Vector3d &v, Eigen::Vector3d &r, Frame from, Frame to);

    private:
      tf::TransformListener tf_listener;

      const std::map<Frame, std::string> frame_ids {
        { Frame::WORLD, "world" },
        { Frame::ODOM, "odom" },
        { Frame::VEHICLE, "vehicle" },
        { Frame::IMU, "vehicle/imu" },
        { Frame::DVL, "vehicle/dvl" },
      };
  };
}
