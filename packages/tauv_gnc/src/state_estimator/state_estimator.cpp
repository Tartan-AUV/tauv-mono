//
// Created by gleb on 5/19/25.
//

#include "tauv_gnc/state_estimator/state_estimator.hpp"

#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/navigation/ImuFactor.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/slam/BetweenFactor.h"

using namespace gtsam;

StateEstimator::StateEstimator() : Node("state_estimator") {
  // ROS subscribers and publishers
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "imu", 10,
      std::bind(&StateEstimator::imu_callback, this, std::placeholders::_1));
  dvl_sub_ = create_subscription<tauv_msgs::msg::WaterlinkedDvlFrame>(
      "dvl", 10,
      std::bind(&StateEstimator::dvl_callback, this, std::placeholders::_1));
  depth_sub_ = create_subscription<tauv_msgs::msg::Depth>(
      "depth", 10,
      std::bind(&StateEstimator::depth_callback, this, std::placeholders::_1));
  odometry_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 10);

  // FLS configuration
  isam_params_ = std::make_shared<gtsam::ISAM2Params>();
  isam_params_->relinearizeThreshold = declare_parameter<double>("relinearize_threshold", 0.0);
  isam_params_->relinearizeSkip = declare_parameter<int>("relinearize_skip", 1);
  lag_ = declare_parameter<double>("lag", 5.0);
  prior_orientation_sigmas_ = declare_parameter<Vector3>("prior_orientation_sigmas", Vector3(0.1, 0.1, 0.1));
  prior_velocity_sigmas_ = declare_parameter<Vector3>("prior_velocity_sigmas", Vector3(0.1, 0.1, 0.1));
  double accelerometer_sigma_ = declare_parameter<double>("accelerometer_sigma", 0.001);
  double gyroscope_sigma_ = declare_parameter<double>("gyroscope_sigma", 0.001);
  imu_preint_params_ = std::make_shared<PreintegrationParams>(Vector3(0.0, 0.0, 9.81));
  imu_preint_params_->setAccelerometerCovariance(I_3x3 * pow(accelerometer_sigma_,2));
  imu_preint_params_->setGyroscopeCovariance(I_3x3 * pow(gyroscope_sigma_,2));
  pim_ = std::make_shared<PreintegratedImuMeasurements>(imu_preint_params_);

  state_ = EstimatorState::AWAITING_PRIOR;
}

void StateEstimator::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  // Initial guess
  // Reset preintegrator with the latest bias
  NavState prev_nav_state = values_.at<NavState>(Symbol('x', k_ - 1));
  imuBias::ConstantBias prev_bias = values_.at<imuBias::ConstantBias>(Symbol('b', k_ - 1));
  pim_->resetIntegrationAndSetBias(prev_bias);
  auto predicted_nav_state = pim_->predict(prev_nav_state, prev_bias);
  values_.insert(Symbol('x', k_), predicted_nav_state);
  values_.insert(Symbol('b', k_), prev_bias);
  auto t_k = rclcpp::Time(msg->header.stamp).seconds();
  timestamps_[Symbol('x', k_)] = t_k;
  timestamps_[Symbol('b', k_)] = t_k;

  // Factor
  auto t_km1 = timestamps_[Symbol('x', k_ - 1)];
  auto dt = t_k - t_km1;
  pim_->integrateMeasurement(Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
                             Vector3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                             dt);
  graph_.emplace_shared<ImuFactor>(
    Symbol('x', k_ - 1),
    Symbol('x', k_),
    Symbol('b', k_ - 1),
    pim_
  );

  auto bias_noise = noiseModel::Diagonal::Sigmas(imu_bias_sigmas_);
  graph_.emplace_shared<BetweenFactor<imuBias::ConstantBias>>(
    Symbol('b', k_), Symbol('b', k_ - 1), imuBias::ConstantBias(), bias_noise
  );

  
}

void StateEstimator::dvl_callback(
    const tauv_msgs::msg::WaterlinkedDvlFrame::SharedPtr msg) {
  // TODO: Implement DVL callback
}

void StateEstimator::depth_callback(
    const tauv_msgs::msg::Depth::SharedPtr msg) {
  if (state_ == EstimatorState::AWAITING_PRIOR) {
    initialize_estimator(msg->depth, msg->variance, msg->header.stamp);
  }
}

void StateEstimator::initialize_estimator(double init_depth,
                                          double init_depth_var,
                                          const rclcpp::Time& timestamp) {
  NavState nav_state_mean{
    Rot3::Identity(),
    Point3(0.0, 0.0, init_depth),
    Velocity3(0.0, 0.0, 0.0)
  };
  Vector9 nav_state_sigmas;
  nav_state_sigmas << prior_orientation_sigmas_, Vector3(0.0, 0.0, init_depth_var), prior_velocity_sigmas_;
  auto nav_state_noise = noiseModel::Diagonal::Sigmas(nav_state_sigmas);
  auto bias_noise = noiseModel::Diagonal::Sigmas(imu_bias_sigmas_);

  Key prior_nav_state_key = Symbol('x', k_);
  Key prior_imu_bias_key = Symbol('b', k_++);

  graph_.addPrior(prior_nav_state_key, nav_state_mean, nav_state_noise);
  graph_.addPrior(prior_imu_bias_key, prior_imu_bias_, bias_noise);
  values_.insert(prior_nav_state_key, nav_state_mean);
  values_.insert(prior_imu_bias_key, prior_imu_bias_);
  timestamps_[prior_nav_state_key] = timestamp.seconds();
  timestamps_[prior_imu_bias_key] = timestamp.seconds();

  state_ = EstimatorState::RUNNING;
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StateEstimator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
