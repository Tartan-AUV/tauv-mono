//
// Created by gleb on 5/19/25.
//

#ifndef STATEESTIMATION_H
#define STATEESTIMATION_H

#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/Expression.h>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tauv_msgs/msg/depth.hpp>
#include <tauv_msgs/msg/waterlinked_dvl_frame.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tuple>
#include <map>

using namespace gtsam;

class StateEstimator final : public rclcpp::Node {
public:
  StateEstimator();

 private:
  void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg);
  void dvl_callback(tauv_msgs::msg::WaterlinkedDvlFrame::SharedPtr msg);
  void depth_callback(tauv_msgs::msg::Depth::SharedPtr msg);
  void initialize_estimator(double init_depth, double init_depth_var,
                            const rclcpp::Time& timestamp);
  void publish_odom(const Vector3& omega);

  // ROS pubs and subs
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<tauv_msgs::msg::WaterlinkedDvlFrame>::SharedPtr dvl_sub_;
  rclcpp::Subscription<tauv_msgs::msg::Depth>::SharedPtr depth_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

  // configurable parameters
  std::shared_ptr<ISAM2Params> isam_params_;
  Vector3 prior_orientation_sigmas_;
  Vector3 prior_velocity_sigmas_;
  imuBias::ConstantBias prior_imu_bias_;
  std::shared_ptr<PreintegratedCombinedMeasurements::Params> preint_params_;
  std::shared_ptr<PreintegratedCombinedMeasurements> preint_measurements_;
  double depth_diff_limit_; // todo: this should be reduced to < 1 / depth_sensor_rate
  double dvl_diff_limit_;

  // ISAM2 objects
  ISAM2 isam_;
  uint64_t k_ = 0;
  double last_update_time_ = 0.0;  // Time of last state update (when DVL measurement arrived)
  double last_imu_time_ = 0.0;     // Time of last IMU measurement for dt calculation
  std::map<uint64_t, double> key_timestamps_;  // Map from key to timestamp
  Vector3 last_imu_omega_ = Vector3::Zero();  // Most recent IMU angular velocity

  // Estimator state
  enum class EstimatorState {
    AWAITING_PRIOR = 0,
    RUNNING,
    Count
  };
  EstimatorState state_;

  // Priors
  Pose3 prior_pose_;
  Vector3 prior_velocity_;
  imuBias::ConstantBias prior_bias_;

  // Utility functions
  Key find_closest_key(double query_t);

  // Transform helper functions
  Vector3 transform_dvl_velocity(const Vector3& dvl_velocity, const rclcpp::Time& timestamp);
  double transform_depth(double depth_value, const rclcpp::Time& timestamp);
  std::string get_frame_name(const std::string& base_frame);

  // TF2 for sensor frame transformations
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string frame_prefix_;
};

#endif //STATEESTIMATION_H
