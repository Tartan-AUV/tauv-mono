#include "state_estimator_ekf.hpp"

#include <manif/manif.h>

#include <tauv_common/constants.hpp>
#include <tauv_common/geometry.hpp>
#include <tauv_common/math_utils.hpp>

#include "../../include/tauv_gnc/state_estimator/state_estimator.hpp"

using namespace tauv::constants;
using namespace tauv::geometry;
using namespace tauv::math_utils;
using namespace std::chrono_literals;
using namespace std::placeholders;
using namespace manif;

StateEstimatorEkf::StateEstimatorEkf() : Node("state_estimator_ekf") {
  // ROS subscribers and publishers
  auto qos = rclcpp::QoS(10);
  imu_sub_.subscribe(this, "imu", qos.get_rmw_qos_profile());
  depth_sub_.subscribe(this, "depth", qos.get_rmw_qos_profile());
  synchronizer_ = std::make_shared<SynchronizerT>(
    ApproximateTimeT(10), imu_sub_, depth_sub_);
  synchronizer_->registerCallback(
    std::bind(&StateEstimatorEkf::imu_depth_callback, this, _1, _2));

  dvl_sub_ = create_subscription<DvlMsg>(
      "dvl", qos, std::bind(&StateEstimatorEkf::dvl_callback, this, _1));

  odometry_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 10);

  // Initialize TF2
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Parameters
  body_frame_ = declare_parameter<std::string>("body_frame");
  depth_frame_ = declare_parameter<std::string>("depth_frame");
  dvl_frame_ = declare_parameter<std::string>("dvl_frame");

  initial_position_stddev_m_ =
      declare_parameter("initial_position_stddev_m", 0.01);
  initial_velocity_stddev_mps_ =
      declare_parameter("initial_velocity_stddev_mps", 0.1);

  // Continuous-time process noise
  double process_noise_density_pos =
    declare_parameter("process_noise_density_pos_m_per_sqrt_s", 0.001);
  double process_noise_density_vel =
    declare_parameter("process_noise_density_vel_mps_per_sqrt_s", 0.001);
  Qc_.setZero();
  Qc_.block(0, 0, 3, 3) = Matrix3d::Identity() * ipow2(process_noise_density_pos);
  Qc_.block(3, 3, 3, 3) = Matrix3d::Identity() * ipow2(process_noise_density_vel);

  // Value for Irvine, CA
  double g = declare_parameter("g", 9.79596);
  a_g_O_ << 0.0, 0.0, g;

  // Initialize buffer
  size_t history_len = declare_parameter("history_length", 20);
  history_ = std::make_shared<HistoryBufferT>(history_len);

  static_tf_timer_ = this->create_wall_timer(
      100ms, std::bind(&StateEstimatorEkf::get_static_transforms, this));
}

void StateEstimatorEkf::imu_depth_callback(ImuMsg::ConstSharedPtr imu_msg, DepthMsg::ConstSharedPtr depth_msg) {
  if (!received_static_transforms_) {
    return;
  }

  if (!received_prior_) {
    initialize(imu_msg, depth_msg);
    received_prior_ = true;
    return;
  }

  auto msg_timestamp = rclcpp::Time(imu_msg->header.stamp);

  auto odom_q_body = quaternion_msg_to_eigen(imu_msg->orientation);
  auto a_body_B = vector3_msg_to_eigen(imu_msg->linear_acceleration);
  auto w_body_B = vector3_msg_to_eigen(imu_msg->angular_velocity);

  const auto& last_entry = history_->back();
  const auto dt = (msg_timestamp - last_entry.stamp).seconds();

  // Predict using the current IMU measurement as "control"
  auto [x_pred, cov_pred] = predict(
    last_entry.state,
    last_entry.cov,
    a_body_B,
    odom_q_body,
    dt
  );

  // Depth measurement Jacobian
  static const Matrix<double, 1, 6> H_depth {0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

  // Observation
  double z = depth_msg->depth;
  // Observation variance
  double R = depth_msg->variance;

  // Innovation
  double y = z - h_depth(x_pred, odom_q_body);
  // Innovation variance
  double S = H_depth * cov_pred * H_depth.transpose() + R;
  // Kalman gain
  Matrix<double, 6, 1> K = cov_pred * H_depth.transpose() / S;
  // State update
  EkfState x_est = x_pred + K * y;
  // Covariance update
  EkfCov cov_est = (I_6x6 + K * H_depth) * cov_pred;

  history_->push_back( Entry {
    odom_q_body,
    a_body_B,
    w_body_B,
    x_est,
    cov_est,
    msg_timestamp,
  });
}

void StateEstimatorEkf::dvl_callback(DvlMsg::ConstSharedPtr msg){
  // TODO: check validity
  // TODO: convert time_of_validity into ros time and use that
  auto stamp = rclcpp::Time(msg->header.stamp);
  auto v_odom_dvl_V = Vector3d(msg->vx, msg->vy, msg->vz);

  // Find nearest entry in history
  double min_time_diff = std::numeric_limits<double>::max();
  std::shared_ptr<Entry> closest_entry = nullptr;
  for (const Entry& entry : history_) {
    auto time_diff = abs((stamp - entry.stamp).seconds()) {

    }
  }
}

void StateEstimatorEkf::get_static_transforms() {
  assert(!received_static_transforms_);

  auto now = get_clock()->now();
  try {
    r_body_depth_B_ =
        vector3_msg_to_eigen(
          tf_buffer_->lookupTransform(body_frame_, depth_frame_, now)
                      .transform.translation);
    body_T_dvl_ = tf_to_se3(
        tf_buffer_->lookupTransform(body_frame_, dvl_frame_, now).transform);

    // Get the adjoint for the DVl transform
    auto dvl_T_body = body_T_dvl_.inverse();
    Ad_dvl_T_body_ = dvl_T_body.adj();

    // Set DVL measurement Jacobian
    F_dvl_ << Matrix3d::Zero(), Ad_dvl_T_body_.block<3, 3>(0,3 );

    static_tf_timer_.reset();
    received_static_transforms_ = true;
  } catch (const tf2::TransformException& e) {
    RCLCPP_WARN(get_logger(),
                "Error while getting static transforms. Are they published? "
                "Retrying...");
  }
}

void StateEstimatorEkf::initialize(ImuMsg::ConstSharedPtr imu_msg, DepthMsg::ConstSharedPtr depth_msg) {
  assert(received_static_transforms_);

  Entry e;

  e.odom_q_body = quaternion_msg_to_eigen(imu_msg->orientation);
  e.a_body_B = vector3_msg_to_eigen(imu_msg->linear_acceleration);
  e.w_body_B = vector3_msg_to_eigen(imu_msg->angular_velocity);

  auto r_odom_depth_O = Vector3d(0.0, 0.0, depth_msg->depth);
  auto r_depth_body_O = -(e.odom_q_body * r_body_depth_B_);
  auto r_odom_body_O = r_odom_depth_O + r_depth_body_O;
  e.state << r_odom_body_O, Vector3d::Zero();

  double var_r_xy = ipow2(initial_position_stddev_m_);
  double var_r_z = depth_msg->variance;
  double var_v = ipow2(initial_velocity_stddev_mps_);
  e.cov.setZero();
  e.cov.diagonal() << var_r_xy, var_r_xy, var_r_z, Vector3d::Ones() * var_v;

  e.stamp = rclcpp::Time(depth_msg->header.stamp);

}

std::tuple<StateEstimatorEkf::EkfState, StateEstimatorEkf::EkfCov> StateEstimatorEkf::predict(
  const EkfState &x,
  const EkfCov &cov,
  const Vector3d &a_body_B,
  const Quaterniond &odom_q_body,
  const double dt
) const {

  // Prev state
  auto r_body_O = x.segment(0, 3);
  auto v_body_O = x.segment(3, 3);

  // Predict state
  Vector3d a_body_O = odom_q_body * a_body_B;
  Vector3d r_body_k_body_pred_O = v_body_O * dt + 0.5 * a_body_O * ipow2(dt);
  Vector3d r_body_pred_O = r_body_O + r_body_k_body_pred_O;

  Vector3d v_body_pred_O = v_body_O + a_body_O * dt;

  EkfState x_pred;
  x_pred << r_body_pred_O, v_body_pred_O;

  // New covariance
  auto F = Matrix<double, 6, 6>::Identity();
  F.block(0, 3, 3, 3) = dt * Matrix3d::Identity();
  EkfCov cov_pred = F * cov * F.transpose() + Qc_ * dt;

  return std::make_tuple(x_pred, cov_pred);
}

Vector3d StateEstimatorEkf::h_dvl(const EkfState &x,
                                  const Vector3d &omega_body_B,
                                  const Quaterniond &odom_q_body) {
  auto v_body_O = x.segment(0, 3);
  auto v_body_B = odom_q_body.inverse() * v_body_O;

  Vector<double, 6> xi_body_B;
  xi_body_B << v_body_B, omega_body_B;
  auto v_dvl_V = (Ad_dvl_T_body_ * xi_body_B).segment(0, 3);
  return v_dvl_V;
}

double StateEstimatorEkf::h_depth(const EkfState &x,
            const Quaterniond &odom_q_body) {
  auto r_body_depth_O = odom_q_body * r_body_depth_B_;
  auto r_odom_depth_O = x.segment(0, 3) + r_body_depth_O;
  return r_odom_depth_O[2];
}
