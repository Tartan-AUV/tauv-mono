//
// Created by gleb on 5/19/25.
//

#include "tauv_gnc/state_estimator/state_estimator.hpp"

#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/inference/Ordering.h"
#include "gtsam/navigation/CombinedImuFactor.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/nonlinear/ExpressionFactor.h"
#include "gtsam/nonlinear/Marginals.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include <Eigen/Eigenvalues>
#include <sstream>

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::X;  // NavState (x,y,z,r,p,y)
using symbol_shorthand::V;  // Velocity (vx,vy,vz)

StateEstimator::StateEstimator() : Node("state_estimator") {
  // Declare frame prefix parameter
  frame_prefix_ = declare_parameter<std::string>("frame_prefix", "");
  
  // Initialize TF2
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  
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

  // ISAM2 configuration
  isam_params_ = std::make_shared<ISAM2Params>();
  isam_params_->relinearizeThreshold = declare_parameter<double>("relinearize_threshold", 0.0);
  isam_params_->relinearizeSkip = declare_parameter<int>("relinearize_skip", 1);

  // Initialize ISAM2 with configured parameters
  isam_ = ISAM2(*isam_params_);

  prior_orientation_sigmas_ = Vector3(
    declare_parameter<std::vector<double>>("prior_orientation_sigmas", {0.1, 0.1, 0.1}).data());
  prior_velocity_sigmas_ = Vector3(
    declare_parameter<std::vector<double>>("prior_velocity_sigmas", {0.1, 0.1, 0.1}).data()); 

  // White noise for accelerometer and gyroscope (continuous time)
  // Values from MTi-200 datasheet
  double accelerometer_sigma_ = declare_parameter<double>("accelerometer_sigma", 5.89e-4);
  double gyroscope_sigma_ = declare_parameter<double>("gyroscope_sigma", 1.75e-4);

  // Bias random walk
  // Ballpark values for some IMU from GTSAM examples
  double accel_bias_rw_sigma = declare_parameter<double>("accel_bias_rw_sigma", 5.0e-3);
  double gyro_bias_rw_sigma = declare_parameter<double>("gyro_bias_rw_sigma", 1.5e-6);

  // Integration error covariance
  double integration_error_cov = declare_parameter<double>("integration_error_cov", 1e-8);
  
  // Covariance of bias used for pre-integration
  double bias_int_cov = declare_parameter<double>("bias_int_cov", 1e-5);

  // IMU preintegration parameters
  preint_params_ = PreintegrationCombinedParams::MakeSharedD();

  preint_params_->accelerometerCovariance = I_3x3 * pow(accelerometer_sigma_, 2);
  preint_params_->gyroscopeCovariance = I_3x3 * pow(gyroscope_sigma_, 2);

  // Bias random walk
  preint_params_->biasAccCovariance = I_3x3 * pow(accel_bias_rw_sigma, 2);
  preint_params_->biasOmegaCovariance = I_3x3 * pow(gyro_bias_rw_sigma,2);

  preint_params_->integrationCovariance = I_3x3 * integration_error_cov;

  preint_params_->biasAccOmegaInt = I_6x6 * bias_int_cov;

  RCLCPP_INFO(get_logger(), "=== IMU PREINTEGRATION PARAMETERS ===");
  RCLCPP_INFO(get_logger(), "Gravity: [%f, %f, %f]", 
    preint_params_->n_gravity.x(), preint_params_->n_gravity.y(), preint_params_->n_gravity.z());
  RCLCPP_INFO(get_logger(), "Accelerometer sigma: %f", accelerometer_sigma_);
  RCLCPP_INFO(get_logger(), "Gyroscope sigma: %f", gyroscope_sigma_);
  RCLCPP_INFO(get_logger(), "Accel bias random walk sigma: %f", accel_bias_rw_sigma);
  RCLCPP_INFO(get_logger(), "Gyro bias random walk sigma: %f", gyro_bias_rw_sigma);
  RCLCPP_INFO(get_logger(), "Integration error covariance: %f", integration_error_cov);
  RCLCPP_INFO(get_logger(), "Bias integration covariance: %f", bias_int_cov);

  // Initialize preintegrated measurements object
  preint_measurements_ = std::make_shared<PreintegratedCombinedMeasurements>(
    preint_params_,
    prior_imu_bias_
  );

  dvl_diff_limit_ = declare_parameter<double>("dvl_max_time_diff", 0.2);

  state_ = EstimatorState::AWAITING_PRIOR;
}

void StateEstimator::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
  if (state_ != EstimatorState::RUNNING) {
    return;
  }

  auto t_k = rclcpp::Time(msg->header.stamp).seconds();

  if (t_k <= last_update_time_) {
    RCLCPP_WARN(get_logger(), "IMU measurement older than last update, skipping.");
    return;
  }

  auto z_acc = Vector3(
    msg->linear_acceleration.x, 
    msg->linear_acceleration.y, 
    msg->linear_acceleration.z
  );

  auto z_omega = Vector3(
    msg->angular_velocity.x,
    msg->angular_velocity.y,
    msg->angular_velocity.z
  );

  // Store angular velocity for odometry message
  last_imu_omega_ = z_omega;

  // Debug: Print first few IMU measurements to check coordinate frame
  static int imu_count = 0;
  if (imu_count < 10) {
    RCLCPP_INFO(get_logger(), "IMU #%d: acc=[%f, %f, %f], gyro=[%f, %f, %f]", 
      imu_count, z_acc.x(), z_acc.y(), z_acc.z(), z_omega.x(), z_omega.y(), z_omega.z());
    
    // Check if we're getting reasonable gravity readings when stationary
    double acc_magnitude = z_acc.norm();
    RCLCPP_INFO(get_logger(), "IMU #%d: acceleration magnitude = %f (expected ~9.81 when stationary)", 
      imu_count, acc_magnitude);
  }
  imu_count++;

  // Calculate dt between consecutive IMU measurements
  double dt;
  if (last_imu_time_ == 0.0) {
    // First IMU measurement - use a small dt or skip integration
    dt = 0.01; // 10ms default, or could skip this measurement
    RCLCPP_INFO(get_logger(), "First IMU measurement, using default dt: %f", dt);
  } else {
    dt = t_k - last_imu_time_;
    if (dt <= 0) {
      RCLCPP_WARN(get_logger(), "Non-positive IMU dt: %f, skipping measurement", dt);
      return;
    }
  }
  
  // Update last IMU time for next iteration
  last_imu_time_ = t_k;
  
  // Debug: Print preintegration details for problematic cases
  if (dt > 0.2) { // Large time steps might cause issues
    RCLCPP_WARN(get_logger(), "Large IMU dt: %f seconds", dt);
  }
  
  preint_measurements_->integrateMeasurement(z_acc, z_omega, dt);

  // Debug: Print preintegration state every 50 measurements
  static int preint_debug_count = 0;
  if (preint_debug_count++ % 50 == 0) {
    auto preint_combined = dynamic_cast<const PreintegratedCombinedMeasurements&>(*preint_measurements_);
    RCLCPP_INFO(get_logger(), "Preint debug: delta_t=%f, delta_v=[%f, %f, %f], delta_p=[%f, %f, %f]",
      preint_combined.deltaTij(),
      preint_combined.deltaVij().x(), preint_combined.deltaVij().y(), preint_combined.deltaVij().z(),
      preint_combined.deltaPij().x(), preint_combined.deltaPij().y(), preint_combined.deltaPij().z());
  }

  RCLCPP_DEBUG(get_logger(), "IMU measurement integrated");
}

void StateEstimator::dvl_callback(
    const tauv_msgs::msg::WaterlinkedDvlFrame::SharedPtr msg) {
  if (state_ != EstimatorState::RUNNING) {
    return;
  }

  RCLCPP_DEBUG(get_logger(), "DVL measurement received");

  const double timestamp = rclcpp::Time(msg->header.stamp).seconds();
  
  if (timestamp <= last_update_time_) {
    RCLCPP_WARN(get_logger(), "DVL measurement older than last update, skipping.");
    return;
  }

  RCLCPP_INFO(get_logger(), "=== DVL CALLBACK DEBUG INFO ===");
  RCLCPP_INFO(get_logger(), "Current key k_: %lu", k_);
  RCLCPP_INFO(get_logger(), "Previous key k_-1: %lu", k_-1);
  RCLCPP_INFO(get_logger(), "Timestamp: %f, Last update time: %f", timestamp, last_update_time_);

  // Get the preintegrated IMU measurements
  auto preint_imu_combined = 
    dynamic_cast<const PreintegratedCombinedMeasurements&>(*preint_measurements_);

  RCLCPP_INFO(get_logger(), "Preintegrated measurements delta_t: %f", preint_imu_combined.deltaTij());
  RCLCPP_INFO(get_logger(), "Preintegrated delta_R: [%f, %f, %f, %f]", 
    preint_imu_combined.deltaRij().toQuaternion().w(),
    preint_imu_combined.deltaRij().toQuaternion().x(),
    preint_imu_combined.deltaRij().toQuaternion().y(),
    preint_imu_combined.deltaRij().toQuaternion().z());
  RCLCPP_INFO(get_logger(), "Preintegrated delta_v: [%f, %f, %f]", 
    preint_imu_combined.deltaVij().x(),
    preint_imu_combined.deltaVij().y(),
    preint_imu_combined.deltaVij().z());
  RCLCPP_INFO(get_logger(), "Preintegrated delta_p: [%f, %f, %f]", 
    preint_imu_combined.deltaPij().x(),
    preint_imu_combined.deltaPij().y(),
    preint_imu_combined.deltaPij().z());

  // Add IMU factor
  NonlinearFactorGraph new_factors;
  Values new_values;

  CombinedImuFactor imu_factor(
    X(k_ - 1), V(k_ - 1), X(k_), V(k_), 
    B(k_ - 1), B(k_),
    preint_imu_combined
  );
  new_factors.add(imu_factor);

  RCLCPP_INFO(get_logger(), "Added IMU factor connecting keys: X(%lu), V(%lu), X(%lu), V(%lu), B(%lu), B(%lu)",
    k_-1, k_-1, k_, k_, k_-1, k_);

  Pose3 x_km1;
  Vector3 v_km1;
  imuBias::ConstantBias b_km1;
  // Get the previous state estimate
  if (k_ <= 1) {
    // Take priors
    x_km1 = prior_pose_;
    v_km1 = prior_velocity_;
    b_km1 = prior_bias_;
  } else {
    // Get the previous state estimate
    auto current_estimate = isam_.calculateEstimate();
    x_km1 = current_estimate.at<Pose3>(X(k_ - 1));
    v_km1 = current_estimate.at<Vector3>(V(k_ - 1));
    b_km1 = current_estimate.at<imuBias::ConstantBias>(B(k_ - 1));
  }

  auto navstate_km1 = NavState(x_km1, v_km1);

  RCLCPP_INFO(get_logger(), "Previous state X(%lu): pos=[%f, %f, %f], rot=[%f, %f, %f, %f]", 
    k_-1, x_km1.x(), x_km1.y(), x_km1.z(),
    x_km1.rotation().toQuaternion().w(), x_km1.rotation().toQuaternion().x(),
    x_km1.rotation().toQuaternion().y(), x_km1.rotation().toQuaternion().z());
  RCLCPP_INFO(get_logger(), "Previous velocity V(%lu): [%f, %f, %f]", k_-1, v_km1.x(), v_km1.y(), v_km1.z());
  RCLCPP_INFO(get_logger(), "Previous bias B(%lu): acc=[%f, %f, %f], gyro=[%f, %f, %f]", k_-1,
    b_km1.accelerometer().x(), b_km1.accelerometer().y(), b_km1.accelerometer().z(),
    b_km1.gyroscope().x(), b_km1.gyroscope().y(), b_km1.gyroscope().z());
  
  // Predict the current state
  auto navstate_k = preint_measurements_->predict(navstate_km1, b_km1);
  auto x_k_hat = navstate_k.pose();
  auto v_k_hat = navstate_k.v();
  auto b_k_hat = b_km1;

  RCLCPP_INFO(get_logger(), "Predicted state X(%lu): pos=[%f, %f, %f], rot=[%f, %f, %f, %f]", 
    k_, x_k_hat.x(), x_k_hat.y(), x_k_hat.z(),
    x_k_hat.rotation().toQuaternion().w(), x_k_hat.rotation().toQuaternion().x(),
    x_k_hat.rotation().toQuaternion().y(), x_k_hat.rotation().toQuaternion().z());
  RCLCPP_INFO(get_logger(), "Predicted velocity V(%lu): [%f, %f, %f]", k_, v_k_hat.x(), v_k_hat.y(), v_k_hat.z());

  // Add DVL factor
  Expression<Vector3>::BinaryFunction<Pose3, Vector3>::type
    dvl_fn = [](const Pose3& pose, const Vector3& vel,
      MakeOptionalJacobian<Vector3, Pose3>::type H1,
      MakeOptionalJacobian<Vector3, Vector3>::type H2) {
      // Rotate the velocity vector from the world frame to the body frame
      return pose.rotation().transpose() * vel;
    };
  Expression<Pose3> xk_expr(X(k_));
  Expression<Vector3> vk_expr(V(k_));
  Expression<Vector3> dvl_expr(dvl_fn, xk_expr, vk_expr);

  Eigen::Matrix3d cov;
  for(int i = 0; i < 9; ++i) {
    cov(i/3, i%3) = static_cast<double>(msg->covariance[i]);
  }

  RCLCPP_INFO(get_logger(), "DVL covariance matrix:");
  for(int i = 0; i < 3; ++i) {
    RCLCPP_INFO(get_logger(), "[%f, %f, %f]", cov(i,0), cov(i,1), cov(i,2));
  }
  RCLCPP_INFO(get_logger(), "DVL measurement: [%f, %f, %f]", msg->vx, msg->vy, msg->vz);

  // Check for problematic covariance values
  double cov_det = cov.determinant();
  Eigen::Vector3d cov_eigenvals = cov.eigenvalues().real();
  RCLCPP_INFO(get_logger(), "DVL covariance determinant: %f", cov_det);
  RCLCPP_INFO(get_logger(), "DVL covariance eigenvalues: [%f, %f, %f]", 
    cov_eigenvals.x(), cov_eigenvals.y(), cov_eigenvals.z());

  if (cov_det <= 0 || cov_eigenvals.minCoeff() <= 0) {
    RCLCPP_ERROR(get_logger(), "DVL covariance matrix is not positive definite!");
  }

  auto dvl_noise = noiseModel::Gaussian::Covariance(cov);
  auto dvl_z = Vector3(msg->vx, msg->vy, msg->vz);
  
  // Transform DVL velocity from DVL frame to base_link frame
  auto dvl_z_transformed = transform_dvl_velocity(dvl_z, rclcpp::Time(msg->header.stamp));
  
  RCLCPP_INFO(get_logger(), "DVL measurement (DVL frame): [%f, %f, %f]", dvl_z.x(), dvl_z.y(), dvl_z.z());
  RCLCPP_INFO(get_logger(), "DVL measurement (base_link frame): [%f, %f, %f]", 
    dvl_z_transformed.x(), dvl_z_transformed.y(), dvl_z_transformed.z());
  
  auto dvl_factor = ExpressionFactor<Vector3>(dvl_noise, dvl_z_transformed, dvl_expr);
  new_factors.add(dvl_factor);

  RCLCPP_INFO(get_logger(), "Added DVL factor connecting keys: X(%lu), V(%lu)", k_, k_);

  // Add the predicted values
  new_values.insert(X(k_), x_k_hat);
  new_values.insert(V(k_), v_k_hat);
  new_values.insert(B(k_), b_k_hat);

  RCLCPP_INFO(get_logger(), "Current factor graph summary:");
  auto existing_factors = isam_.getFactorsUnsafe();
  RCLCPP_INFO(get_logger(), "Existing factors: %lu", existing_factors.size());
  RCLCPP_INFO(get_logger(), "New factors: %lu", new_factors.size());
  
  // DEBUG: Analyze linear system before calling calculateEstimate()
  try {
    RCLCPP_INFO(get_logger(), "=== LINEAR SYSTEM ANALYSIS ===");
    
    // Get current linearization point
    auto linearization_point = isam_.getLinearizationPoint();
    RCLCPP_INFO(get_logger(), "Variables in linearization point: %lu", linearization_point.size());
    
    // Print variables in linearization point
    for (const auto& key_value : linearization_point) {
      RCLCPP_INFO(get_logger(), "Variable key: %lu", key_value.key);
    }
    
    // Linearize the factor graph
    auto linear_graph = existing_factors.linearize(linearization_point);
    RCLCPP_INFO(get_logger(), "Linearized factor graph size: %lu", linear_graph->size());
    
    // Create an ordering for the linear system
    auto ordering = Ordering::Colamd(*linear_graph);
    RCLCPP_INFO(get_logger(), "Variable ordering (COLAMD):");
    for (size_t i = 0; i < ordering.size(); ++i) {
      auto key = Symbol(ordering[i]);
      RCLCPP_INFO(get_logger(), "Position %lu: Variable key %c_%lu", i, key.chr(), key.index());
    }
    
    // Get Jacobian matrix A and RHS vector b such that the system is A*x = b
    auto jacobian_pair = linear_graph->jacobian(ordering);
    Matrix jacobian_A = jacobian_pair.first;
    Vector jacobian_b = jacobian_pair.second;

    // Print Jacobian matrix values
    RCLCPP_INFO(get_logger(), "=== Jacobian Matrix Values ===");
    for (int i = 0; i < jacobian_A.rows(); ++i) {
      std::stringstream row_values;
      row_values << "Row " << i << ": ";
      for (int j = 0; j < jacobian_A.cols(); ++j) {
        row_values << jacobian_A(i,j) << " ";
      }
      RCLCPP_INFO(get_logger(), "%s", row_values.str().c_str());
    }
    
    // Print RHS vector values
    RCLCPP_INFO(get_logger(), "=== RHS Vector Values ===");
    for (int i = 0; i < jacobian_b.size(); ++i) {
      RCLCPP_INFO(get_logger(), "b[%d] = %f", i, jacobian_b(i));
    }

    RCLCPP_INFO(get_logger(), "Jacobian matrix A size: %d x %d", jacobian_A.rows(), jacobian_A.cols());
    RCLCPP_INFO(get_logger(), "RHS vector b size: %d", jacobian_b.size());
    
    // Check for zero rows/columns in Jacobian
    int zero_rows = 0;
    int zero_cols = 0;
    for (int i = 0; i < jacobian_A.rows(); ++i) {
      if (jacobian_A.row(i).norm() < 1e-12) zero_rows++;
    }
    for (int j = 0; j < jacobian_A.cols(); ++j) {
      if (jacobian_A.col(j).norm() < 1e-12) zero_cols++;
    }
    RCLCPP_INFO(get_logger(), "Near-zero rows in Jacobian A: %d, Near-zero cols: %d", zero_rows, zero_cols);

    
    
    // Get Hessian matrix Lambda and information vector eta such that cost = 0.5*x'*Lambda*x + eta'*x + const
    auto hessian_pair = linear_graph->hessian(ordering);
    Matrix hessian_Lambda = hessian_pair.first;
    Vector hessian_eta = hessian_pair.second;
    
    RCLCPP_INFO(get_logger(), "Hessian matrix Lambda size: %d x %d", hessian_Lambda.rows(), hessian_Lambda.cols());
    RCLCPP_INFO(get_logger(), "Information vector eta size: %d", hessian_eta.size());
    
    // Check Hessian properties
    double hessian_determinant = hessian_Lambda.determinant();
    RCLCPP_INFO(get_logger(), "Hessian determinant: %e", hessian_determinant);
    
    // Check eigenvalues of Hessian for positive definiteness
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(hessian_Lambda);
    auto eigenvalues = eigen_solver.eigenvalues();
    double min_eigenvalue = eigenvalues.minCoeff();
    double max_eigenvalue = eigenvalues.maxCoeff();
    double condition_number = max_eigenvalue / std::max(std::abs(min_eigenvalue), 1e-12);
    
    RCLCPP_INFO(get_logger(), "Hessian eigenvalues - Min: %e, Max: %e", min_eigenvalue, max_eigenvalue);
    RCLCPP_INFO(get_logger(), "Condition number: %e", condition_number);
    
    // Count negative eigenvalues
    int negative_eigenvalues = 0;
    int near_zero_eigenvalues = 0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
      if (eigenvalues(i) < -1e-10) negative_eigenvalues++;
      else if (std::abs(eigenvalues(i)) < 1e-8) near_zero_eigenvalues++;
    }
    
    RCLCPP_INFO(get_logger(), "Negative eigenvalues: %d, Near-zero eigenvalues: %d", 
                negative_eigenvalues, near_zero_eigenvalues);
    
    if (min_eigenvalue <= 0) {
      RCLCPP_WARN(get_logger(), "DETECTED: Hessian is not positive definite! (min eigenvalue: %e)", min_eigenvalue);
    }
    if (condition_number > 1e12) {
      RCLCPP_WARN(get_logger(), "DETECTED: System is poorly conditioned! (condition number: %e)", condition_number);
    }
    if (near_zero_eigenvalues > 0) {
      RCLCPP_WARN(get_logger(), "DETECTED: System has near-zero eigenvalues indicating underconstrained variables!");
    }
    if (zero_cols > 0) {
      RCLCPP_WARN(get_logger(), "DETECTED: Jacobian has zero columns indicating unconstrained variables!");
    }
    
    // Print small Jacobian matrix if it's small enough
    if (jacobian_A.rows() <= 20 && jacobian_A.cols() <= 20) {
      RCLCPP_INFO(get_logger(), "=== JACOBIAN MATRIX A ===");
      std::stringstream ss_A;
      ss_A << jacobian_A;
      RCLCPP_INFO(get_logger(), "Jacobian A:\n%s", ss_A.str().c_str());
      
      RCLCPP_INFO(get_logger(), "=== RHS VECTOR b ===");
      std::stringstream ss_b;
      ss_b << jacobian_b.transpose();
      RCLCPP_INFO(get_logger(), "RHS b: %s", ss_b.str().c_str());
    }
    
    // Print small Hessian matrix if it's small enough  
    if (hessian_Lambda.rows() <= 10 && hessian_Lambda.cols() <= 10) {
      RCLCPP_INFO(get_logger(), "=== HESSIAN MATRIX Lambda ===");
      std::stringstream ss_Lambda;
      ss_Lambda << hessian_Lambda;
      RCLCPP_INFO(get_logger(), "Hessian Lambda:\n%s", ss_Lambda.str().c_str());
      
      RCLCPP_INFO(get_logger(), "=== INFORMATION VECTOR eta ===");
      std::stringstream ss_eta;
      ss_eta << hessian_eta.transpose();
      RCLCPP_INFO(get_logger(), "Information eta: %s", ss_eta.str().c_str());
    }
    
    // Print eigenvalues for debugging
    if (eigenvalues.size() <= 20) {
      RCLCPP_INFO(get_logger(), "=== HESSIAN EIGENVALUES ===");
      std::stringstream ss_eigvals;
      ss_eigvals << eigenvalues.transpose();
      RCLCPP_INFO(get_logger(), "Eigenvalues: %s", ss_eigvals.str().c_str());
    }
    
    RCLCPP_INFO(get_logger(), "=== END LINEAR SYSTEM ANALYSIS ===");
    
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Error during linear system analysis: %s", e.what());
  }
  
  auto existing_values = isam_.calculateEstimate();
  RCLCPP_INFO(get_logger(), "Existing variables: %lu", existing_values.size());
  RCLCPP_INFO(get_logger(), "New variables: %lu", new_values.size());

  // Evaluate the DVL factor error at linearization point
  Vector3 dvl_prediction = x_k_hat.rotation().transpose() * v_k_hat;
  Vector3 dvl_error = dvl_prediction - dvl_z_transformed;
  RCLCPP_INFO(get_logger(), "DVL factor error at linearization point: [%f, %f, %f]", 
    dvl_error.x(), dvl_error.y(), dvl_error.z());
  RCLCPP_INFO(get_logger(), "DVL predicted (body frame): [%f, %f, %f]", 
    dvl_prediction.x(), dvl_prediction.y(), dvl_prediction.z());

  try {
    // Update ISAM2
    RCLCPP_INFO(get_logger(), "Calling ISAM2 update...");
    isam_.update(new_factors, new_values);
    isam_.update();
    RCLCPP_INFO(get_logger(), "ISAM2 update successful");
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "ISAM2 update failed: %s", e.what());
    return;
  }

  // Store timestamp
  key_timestamps_[k_] = timestamp;
  last_update_time_ = timestamp;

  // Reset the preintegrator
  preint_measurements_->resetIntegrationAndSetBias(b_k_hat);

  // Reset IMU time tracking for next preintegration period
  last_imu_time_ = 0.0;

  RCLCPP_INFO(get_logger(), "Reset preintegrator with bias: acc=[%f, %f, %f], gyro=[%f, %f, %f]",
    b_k_hat.accelerometer().x(), b_k_hat.accelerometer().y(), b_k_hat.accelerometer().z(),
    b_k_hat.gyroscope().x(), b_k_hat.gyroscope().y(), b_k_hat.gyroscope().z());

  ++k_;
}

void StateEstimator::depth_callback(
    const tauv_msgs::msg::Depth::SharedPtr msg) {
  if (state_ == EstimatorState::AWAITING_PRIOR) {
    // Transform depth from depth sensor frame to base_link frame
    double transformed_depth = transform_depth(msg->depth, rclcpp::Time(msg->header.stamp));
    
    RCLCPP_INFO(get_logger(), "Depth measurement (depth frame): %f", msg->depth);
    RCLCPP_INFO(get_logger(), "Depth measurement (base_link frame): %f", transformed_depth);
    
    initialize_estimator(transformed_depth, msg->variance, msg->header.stamp);
  }
}

void StateEstimator::initialize_estimator(double init_depth,
                                          double init_depth_var,
                                          const rclcpp::Time& timestamp) {
  if (state_ != EstimatorState::AWAITING_PRIOR) {
    RCLCPP_WARN(get_logger(), "Trying to initialize a running estimator!");
    return;
  }

  RCLCPP_DEBUG(get_logger(), "Initializing the estimator");

  RCLCPP_INFO(get_logger(), "=== ESTIMATOR INITIALIZATION DEBUG INFO ===");
  RCLCPP_INFO(get_logger(), "Initial depth: %f, variance: %f", init_depth, init_depth_var);

  // Prior state
  auto R_prior = Rot3::Identity();
  auto r_prior = Point3(0.0, 0.0, init_depth);
  auto T_prior = Pose3(R_prior, r_prior);
  auto v_prior = Vector3(0.0, 0.0, 0.0);
  auto b_prior = prior_imu_bias_;
  
  // Store prior values as member variables
  prior_pose_ = T_prior;
  prior_velocity_ = v_prior;
  prior_bias_ = b_prior;
  
  RCLCPP_INFO(get_logger(), "Prior pose: pos=[%f, %f, %f], rot=Identity", r_prior.x(), r_prior.y(), r_prior.z());
  RCLCPP_INFO(get_logger(), "Prior velocity: [%f, %f, %f]", v_prior.x(), v_prior.y(), v_prior.z());
  RCLCPP_INFO(get_logger(), "Prior bias: acc=[%f, %f, %f], gyro=[%f, %f, %f]",
    b_prior.accelerometer().x(), b_prior.accelerometer().y(), b_prior.accelerometer().z(),
    b_prior.gyroscope().x(), b_prior.gyroscope().y(), b_prior.gyroscope().z());
  
  // Prior state covariance
  auto T_prior_noise = noiseModel::Diagonal::Sigmas(
    (Vector(6) << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1).finished()
  );
  auto v_prior_noise = noiseModel::Isotropic::Sigma(3, 0.1);
  auto b_prior_noise = noiseModel::Isotropic::Sigma(6, 1e-3);

  RCLCPP_INFO(get_logger(), "Prior pose noise sigmas: [0.01, 0.01, 0.01, 0.1, 0.1, 0.1] (x,y,z,rx,ry,rz)");
  RCLCPP_INFO(get_logger(), "Prior velocity noise sigma: 0.1");
  RCLCPP_INFO(get_logger(), "Prior bias noise sigma: 1e-3");

  // Initial values and factors
  Values values;
  values.insert(X(k_), T_prior);
  values.insert(V(k_), v_prior);
  values.insert(B(k_), b_prior);

  NonlinearFactorGraph graph;
  graph.addPrior(X(k_), T_prior, T_prior_noise);
  graph.addPrior(V(k_), v_prior, v_prior_noise);
  graph.addPrior(B(k_), b_prior, b_prior_noise);

  RCLCPP_INFO(get_logger(), "Added %lu prior factors for key %lu", graph.size(), k_);

  // Initialize ISAM2
  isam_.update(graph, values);

  // Store timestamp
  key_timestamps_[k_] = timestamp.seconds();
  last_update_time_ = timestamp.seconds();

  // Increment the key counter
  ++k_;

  state_ = EstimatorState::RUNNING;

  RCLCPP_INFO(get_logger(), "State estimator initialized with a depth prior.");
}

Key StateEstimator::find_closest_key(double query_t) {
  double best_diff = std::numeric_limits<double>::infinity();
  Key best_key = 0;
  
  for (const auto& [key, t] : key_timestamps_) {
    double diff = std::abs(t - query_t);
    if (diff < best_diff) {
      best_diff = diff;
      best_key = key;
    }
  }

  return best_key;
}

void StateEstimator::publish_odom(const Vector3& omega) {
  Values current_estimate = isam_.calculateBestEstimate();

  auto x_k = current_estimate.at<Pose3>(X(k_));
  auto x_k_p = x_k.translation();
  auto x_k_q = x_k.rotation().toQuaternion();
  auto v_k = current_estimate.at<Velocity3>(V(k_));

  Marginals marginals(isam_.getFactorsUnsafe(), current_estimate);
  auto x_k_cov = marginals.marginalCovariance(X(k_));
  auto v_k_cov = marginals.marginalCovariance(V(k_));

  nav_msgs::msg::Odometry odom;
  odom.header.stamp = now();
  odom.header.frame_id = "odom";
  odom.child_frame_id = "base_link";

  geometry_msgs::msg::Point msg_p;
  msg_p.x = x_k_p.x();
  msg_p.y = x_k_p.y();
  msg_p.z = x_k_p.z();
  odom.pose.pose.position = msg_p;

  geometry_msgs::msg::Quaternion msg_q;
  msg_q.w = x_k_q.w();
  msg_q.x = x_k_q.x();
  msg_q.y = x_k_q.y();
  msg_q.z = x_k_q.z();
  odom.pose.pose.orientation = msg_q;

  for(int i=0;i<6;i++)
    for(int j=0;j<6;j++)
      odom.pose.covariance[i*6+j] = x_k_cov(i,j);

  odom.twist.twist.linear.x = v_k.x();
  odom.twist.twist.linear.y = v_k.y();
  odom.twist.twist.linear.z = v_k.z();
  odom.twist.twist.angular.x = omega[0];
  odom.twist.twist.angular.y = omega[1];
  odom.twist.twist.angular.z = omega[2];
  for(int i=0;i<6;i++)
    for(int j=0;j<6;j++)
      odom.twist.covariance[i*6+j] = (i<3&&j<3) ? v_k_cov(i,j) : 0.0;

  odometry_pub_->publish(odom);
}

std::string StateEstimator::get_frame_name(const std::string& base_frame) {
  if (frame_prefix_.empty()) {
    return base_frame;
  }
  return frame_prefix_ + "/" + base_frame;
}

Vector3 StateEstimator::transform_dvl_velocity(const Vector3& dvl_velocity, const rclcpp::Time& timestamp) {
  try {
    std::string base_link_frame = get_frame_name("base_link");
    std::string dvl_frame = get_frame_name("dvl");
    
    // Get transform from dvl frame to base_link
    geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
      base_link_frame, dvl_frame, timestamp, rclcpp::Duration::from_nanoseconds(50000000)); // 50ms timeout
    
    // Create velocity vector in dvl frame
    geometry_msgs::msg::Vector3Stamped dvl_vel_msg;
    dvl_vel_msg.header.frame_id = dvl_frame;
    dvl_vel_msg.header.stamp = timestamp;
    dvl_vel_msg.vector.x = dvl_velocity.x();
    dvl_vel_msg.vector.y = dvl_velocity.y();
    dvl_vel_msg.vector.z = dvl_velocity.z();
    
    // Transform to base_link frame
    geometry_msgs::msg::Vector3Stamped base_vel_msg;
    tf2::doTransform(dvl_vel_msg, base_vel_msg, transform);
    
    return Vector3(base_vel_msg.vector.x, base_vel_msg.vector.y, base_vel_msg.vector.z);
    
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(get_logger(), "Could not transform DVL velocity: %s", ex.what());
    // Return the original velocity if transform fails
    return dvl_velocity;
  }
}

double StateEstimator::transform_depth(double depth_value, const rclcpp::Time& timestamp) {
  try {
    std::string base_link_frame = get_frame_name("base_link");
    std::string depth_frame = get_frame_name("depth");
    
    // Get transform from depth frame to base_link
    geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
      base_link_frame, depth_frame, timestamp, rclcpp::Duration::from_nanoseconds(50000000)); // 50ms timeout
    
    // For depth, we primarily care about the Z-offset between frames
    // The depth sensor measures distance, but we need to account for its offset from base_link
    double depth_offset_z = transform.transform.translation.z;
    
    // Adjust depth by the Z-offset (positive Z is up in ROS convention)
    // If depth sensor is below base_link (negative Z), the actual depth at base_link is less
    return depth_value - depth_offset_z;
    
  } catch (tf2::TransformException& ex) {
    RCLCPP_WARN(get_logger(), "Could not transform depth measurement: %s", ex.what());
    // Return the original depth if transform fails
    return depth_value;
  }
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StateEstimator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
