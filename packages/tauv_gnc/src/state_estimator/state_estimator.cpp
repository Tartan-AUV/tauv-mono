//
// Created by gleb on 5/19/25.
//

#include "tauv_gnc/state_estimator/state_estimator.hpp"

#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/navigation/ImuFactor.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/nonlinear/ExpressionFactor.h"
#include "gtsam/nonlinear/Marginals.h"
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
  prior_orientation_sigmas_ = Vector3(
    declare_parameter<std::vector<double>>("prior_orientation_sigmas", {0.1, 0.1, 0.1}).data());
  prior_velocity_sigmas_ = Vector3(
    declare_parameter<std::vector<double>>("prior_velocity_sigmas", {0.1, 0.1, 0.1}).data());
  double accelerometer_sigma_ = declare_parameter<double>("accelerometer_sigma", 0.001);
  double gyroscope_sigma_ = declare_parameter<double>("gyroscope_sigma", 0.001);
  auto prior_imu_bias_vec = declare_parameter<std::vector<double>>("prior_imu_bias",
{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  prior_imu_bias_ = imuBias::ConstantBias(Vector6(prior_imu_bias_vec.data()));
  auto imu_bias_sigmas_vec = declare_parameter<std::vector<double>>("imu_bias_sigmas", {1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3});
  imu_bias_sigmas_ = Vector6(imu_bias_sigmas_vec.data());
  imu_preint_params_ = std::make_shared<PreintegrationParams>(Vector3(0.0, 0.0, 9.81));
  imu_preint_params_->setAccelerometerCovariance(I_3x3 * pow(accelerometer_sigma_,2));
  imu_preint_params_->setGyroscopeCovariance(I_3x3 * pow(gyroscope_sigma_,2));
  pim_ = std::make_shared<PreintegratedImuMeasurements>(imu_preint_params_);
  depth_diff_limit_ = declare_parameter<double>("depth_max_time_diff", 0.02);
  dvl_diff_limit_ = declare_parameter<double>("dvl_max_time_diff", 0.2);

  state_ = EstimatorState::AWAITING_PRIOR;
}

void StateEstimator::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {

  if (state_ != EstimatorState::RUNNING) {
    return;
  }

  // Initial guess
  // Reset preintegrator with the latest bias

  Key key_xkm1 = Symbol('x', k_ - 1);
  Key key_xk = Symbol('x', k_);
  Key key_bkm1 = Symbol('b', k_ - 1);
  Key key_bk = Symbol('b', k_);

  auto t_k = rclcpp::Time(msg->header.stamp).seconds();

  auto t_km1 = smoother_.timestamps().at(key_xkm1);
  auto dt = t_k - t_km1;

  if (dt <= 0.0) {
    RCLCPP_WARN(get_logger(), "IMU measurement dt < 0.0, skipping.");
    return;
  }

  NavState xkm1 = smoother_.calculateEstimate<NavState>(key_xkm1);
  imuBias::ConstantBias bkm1 = smoother_.calculateEstimate<imuBias::ConstantBias>(key_bkm1);

  pim_->resetIntegrationAndSetBias(bkm1);

  NavState xk_hat = pim_->predict(xkm1, bkm1);
  imuBias::ConstantBias bk_hat = bkm1;

  NonlinearFactorGraph new_factors;
  Values new_values;
  FixedLagSmootherKeyTimestampMap new_timestamps;

  new_values.insert(key_xk, xk_hat);
  new_values.insert(key_bk, bk_hat);

  new_timestamps[key_xk] = t_k;
  new_timestamps[key_bk] = t_k;

  auto z_acc = Vector3(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
  auto z_omega = Vector3(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

  pim_->integrateMeasurement(z_acc, z_omega, dt);

  new_factors.emplace_shared<ImuFactor2>(
    Symbol('x', k_ - 1),
    Symbol('x', k_),
    Symbol('b', k_ - 1),
    *pim_
  );

  auto bias_noise = noiseModel::Diagonal::Sigmas(imu_bias_sigmas_);
  new_factors.emplace_shared<BetweenFactor<imuBias::ConstantBias>>(
    Symbol('b', k_), Symbol('b', k_ - 1), imuBias::ConstantBias(), bias_noise
  );

  auto result = smoother_.update(new_factors, new_values, new_timestamps);

  publish_odom(z_omega);

  ++k_;
}

void StateEstimator::dvl_callback(
    const tauv_msgs::msg::WaterlinkedDvlFrame::SharedPtr msg) {

  if (state_ != EstimatorState::RUNNING) {
    return;
  }

  const double timestamp = rclcpp::Time(msg->header.stamp).seconds(); // this is "time of validity" of the dvl measurement
  const auto [key_xk, diff] = find_closest_timestamp(timestamp, 'x');

  Expression<Vector3>::UnaryFunction<NavState>::type
    dvl_fn = [](const NavState& ns,
      MakeOptionalJacobian<Vector3, NavState>::type) {
      return ns.pose().rotation().transpose() * ns.velocity();
    };
  Expression<NavState> xk_expr(key_xk);
  Expression<Vector3> dvl_expr(dvl_fn, xk_expr);

  Eigen::Matrix3d cov;
  for(int i = 0; i < 9; ++i) {
    cov(i/3, i%3) = static_cast<double>(msg->covariance[i]);
  }

  auto dvl_noise = noiseModel::Gaussian::Covariance(cov);

  auto dvl_z = Vector3(msg->vx, msg->vy, msg->vz);

  auto dvl_factor = ExpressionFactor<Vector3>(dvl_noise, dvl_z, dvl_expr);

  NonlinearFactorGraph new_factors;
  new_factors.push_back(dvl_factor);

  smoother_.update(new_factors);
}

void StateEstimator::depth_callback(
    const tauv_msgs::msg::Depth::SharedPtr msg) {

  if (state_ == EstimatorState::AWAITING_PRIOR) {
    initialize_estimator(msg->depth, msg->variance, msg->header.stamp);
    return;
  }

  const double timestamp = rclcpp::Time(msg->header.stamp).seconds();
  const auto [key_xk, diff] = find_closest_timestamp(timestamp, 'x');
  //  TODO CHECK diff
  Expression<double>::UnaryFunction<NavState>::type
      depth_fn = [](const NavState& ns, MakeOptionalJacobian<double, NavState>::type)
      {
        return ns.pose().translation().z();
      };
  Expression<NavState> xk_expr(key_xk);
  Expression<double> depth_expr(depth_fn, xk_expr);

  auto depth_noise = noiseModel::Isotropic::Sigma(1, 1e-3);

  auto depth_factor = ExpressionFactor(depth_noise, msg->depth, depth_expr);

  NonlinearFactorGraph new_factors;
  new_factors.push_back(depth_factor);

  smoother_.update(new_factors);
}

void StateEstimator::initialize_estimator(double init_depth,
                                          double init_depth_var,
                                          const rclcpp::Time& timestamp) {
  if (state_ != EstimatorState::AWAITING_PRIOR) {
    RCLCPP_WARN(get_logger(), "Trying to initialize a running estimator!");
    return;
  }
  NavState nav_state_mean{
    Rot3::Identity(),
    Point3(0.0, 0.0, init_depth),
    Velocity3(0.0, 0.0, 0.0)
  };
  Vector9 nav_state_sigmas;
  // nav_state_sigmas << prior_orientation_sigmas_, Vector3(1e-3, 1e-3, std::max(init_depth_var, 1e-3)), prior_velocity_sigmas_;
  auto nav_state_noise = noiseModel::Isotropic::Sigma(9, 1e-3);
  RCLCPP_INFO(get_logger(),
            "bias sigmas = [%g %g %g %g %g %g]",
            imu_bias_sigmas_[0], imu_bias_sigmas_[1],
            imu_bias_sigmas_[2], imu_bias_sigmas_[3],
            imu_bias_sigmas_[4], imu_bias_sigmas_[5]);
  auto bias_noise = noiseModel::Diagonal::Sigmas(imu_bias_sigmas_);

  double time = timestamp.seconds();

  Key key_x0 = Symbol('x', k_);
  Key key_b0 = Symbol('b', k_);

  NonlinearFactorGraph new_factors;
  Values new_values;
  FixedLagSmootherKeyTimestampMap new_timestamps;

  new_factors.addPrior(key_x0, nav_state_mean, nav_state_noise); // maybe should be add(PriorFactor)
  new_factors.addPrior(key_b0, prior_imu_bias_, bias_noise);
  new_values.insert(key_x0, nav_state_mean);
  new_values.insert(key_b0, prior_imu_bias_);
  new_timestamps[key_x0] = time;
  new_timestamps[key_b0] = time;

  smoother_.update(new_factors, new_values, new_timestamps);

  ++k_;

  std::cout << time << "\n";

  RCLCPP_INFO(get_logger(), "State estimator initialized with a depth prior.");

  state_ = EstimatorState::RUNNING;
}

std::tuple<Key, double> StateEstimator::find_closest_timestamp(double query_t,
                                                               char symbol) {
  double best_diff = std::numeric_limits<double>::infinity();
  Key best_key = 0;
  auto ktm = smoother_.timestamps();
  for (auto const& [key, t] : ktm) {
    if (Symbol(key).chr() == symbol) {
      double diff = std::abs(t - query_t);
      if (diff < best_diff) {
        best_diff = diff;
        best_key = key;
      }
    }
  }

  return std::make_tuple(best_key, best_diff);
}

void StateEstimator::publish_odom(const Vector3& omega) {

  auto values = smoother_.calculateEstimate();
  Key key_xk = Symbol('x', k_);

  const NavState xk = values.at<NavState>(key_xk);

  Marginals marginals(smoother_.getFactors(), values);
  auto fullCov = marginals.marginalCovariance(key_xk);
  // pose covariance is top-left 6×6, vel covariance is bottom-right 3×3
  Eigen::Matrix<double,6,6> poseCov = fullCov.topLeftCorner<6,6>();
  Eigen::Matrix3d velCov   = fullCov.block<3,3>(6,6);

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = now();
  odom.header.frame_id    = "odom";
  odom.child_frame_id     = "base_link";

  auto xk_p = xk.pose().translation();
  auto xk_q = xk.pose().rotation().toQuaternion();

  geometry_msgs::msg::Point msg_p;
  msg_p.x = xk_p.x();
  msg_p.y = xk_p.y();
  msg_p.z = xk_p.z();
  odom.pose.pose.position = msg_p;

  geometry_msgs::msg::Quaternion msg_q;
  msg_q.w = xk_q.w();
  msg_q.x = xk_q.x();
  msg_q.y = xk_q.y();
  msg_q.z = xk_q.z();
  odom.pose.pose.orientation = msg_q;

  for(int i=0;i<6;i++)
    for(int j=0;j<6;j++)
      odom.pose.covariance[i*6+j] = poseCov(i,j);

  auto v = xk.v();
  odom.twist.twist.linear.x = v.x();
  odom.twist.twist.linear.y = v.y();
  odom.twist.twist.linear.z = v.z();
  odom.twist.twist.angular.x = omega[0];
  odom.twist.twist.angular.y = omega[1];
  odom.twist.twist.angular.z = omega[2];
  for(int i=0;i<6;i++)
    for(int j=0;j<6;j++)
      odom.twist.covariance[i*6+j] = (i<3&&j<3) ? velCov(i,j) : 0.0;

  odometry_pub_->publish(odom);
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StateEstimator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
