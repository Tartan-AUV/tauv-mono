//
// Created by gleb on 5/19/25.
//

#ifndef STATEESTIMATION_H
#define STATEESTIMATION_H

#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tauv_msgs/msg/depth.hpp>
#include <tauv_msgs/msg/waterlinked_dvl_frame.hpp>
#include <tuple>

#include "gtsam/navigation/ImuBias.h"
#include "gtsam/navigation/PreintegrationParams.h"
#include "gtsam/nonlinear/Expression.h"

namespace gtsam {
class PreintegratedImuMeasurements;
}
class StateEstimator final : public rclcpp::Node {
public:
  StateEstimator();

 private:
  void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg);
  void dvl_callback(tauv_msgs::msg::WaterlinkedDvlFrame::SharedPtr msg);
  void depth_callback(tauv_msgs::msg::Depth::SharedPtr msg);
  void initialize_estimator(double init_depth, double init_depth_var,
                            const rclcpp::Time& timestamp);
  void publish_odom(const gtsam::Vector3& omega);


  // ROS pubs and subs
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<tauv_msgs::msg::WaterlinkedDvlFrame>::SharedPtr dvl_sub_;
  rclcpp::Subscription<tauv_msgs::msg::Depth>::SharedPtr depth_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

  // configurable parameters
  std::shared_ptr<gtsam::ISAM2Params> isam_params_;
  double lag_;
  gtsam::Vector3 prior_orientation_sigmas_;
  gtsam::Vector3 prior_velocity_sigmas_;
  gtsam::imuBias::ConstantBias prior_imu_bias_;
  gtsam::Vector6 imu_bias_sigmas_;
  std::shared_ptr<gtsam::PreintegrationParams> imu_preint_params_;
  std::shared_ptr<gtsam::PreintegratedImuMeasurements> pim_;
  double depth_diff_limit_; // todo: this should be reduced to < 1 / depth_sensor_rate
  double dvl_diff_limit_;

  // FLS objects
  gtsam::IncrementalFixedLagSmoother smoother_;
  uint64_t k_ = 0;

  // Estimator state
  enum class EstimatorState {
    AWAITING_PRIOR = 0,
    RUNNING,
    Count
  };
  EstimatorState state_;

  // Utility functions
  std::tuple<gtsam::Key, double> find_closest_timestamp(double query_t,
                                                        char symbol);

};



#endif //STATEESTIMATION_H
