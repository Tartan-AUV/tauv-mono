#pragma once

#include <manif/manif.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/circular_buffer.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/time_synchronizer.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <string>
#include <tauv_msgs/msg/depth.hpp>
#include <tauv_msgs/msg/waterlinked_dvl_frame.hpp>
#include <tuple>

using manif::SE_2_3d;
using manif::SE3d;
using namespace Eigen;
using DepthMsg = tauv_msgs::msg::Depth;
using DvlMsg = tauv_msgs::msg::WaterlinkedDvlFrame;
using ImuMsg = sensor_msgs::msg::Imu;

class StateEstimatorEkf : public rclcpp::Node {

public:
  StateEstimatorEkf();

private:
  /* Estimator state */
  // The odometry frame origin is on the pool surface, and the xy-plane is level
  // Order: translation (3), velocity (3)
  using EkfState = Vector<double, 6>;
  /* Estiamtor covariance */
  using EkfCov = Matrix<double, 6, 6>;

  struct Entry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Quaterniond odom_q_body;
    Vector3d a_body_B;
    Vector3d w_body_B;
    EkfState state;
    EkfCov cov;
    rclcpp::Time stamp;
  };


  /* Callbacks */
  void imu_depth_callback(ImuMsg::ConstSharedPtr imu_msg, DepthMsg::ConstSharedPtr depth_msg);
  void dvl_callback(DvlMsg::ConstSharedPtr msg);

  void get_static_transforms();
  void initialize(ImuMsg::ConstSharedPtr, DepthMsg::ConstSharedPtr depth_msg);

  std::tuple<EkfState, EkfCov> predict(
    const EkfState &x,
    const EkfCov &cov,
    const Vector3d &a_body_B,
    const Quaterniond &odom_q_body,
    double dt
  ) const;

  Vector3d h_dvl(const EkfState &x, const Vector3d &omega_body_B,
                     const Quaterniond &odom_q_body);
  double h_depth(const EkfState &x, const Quaterniond &odom_q_body);

  // History buffer
  using HistoryBufferT = boost::circular_buffer<Entry, aligned_allocator<Entry>>;
  std::shared_ptr<HistoryBufferT> history_;

  // Whether we have received the prior;
  bool received_prior_ = false;

  /* Parameter values */
  // TF frames
  std::string body_frame_;
  std::string dvl_frame_;
  std::string depth_frame_;

  // Initial state uncertainty
  double initial_position_stddev_m_;
  double initial_velocity_stddev_mps_;

  // (Measurement noise is taken from messages)

  // Process noise
  Matrix<double, 6, 6> Qc_;

  // Specific force due to gravity (pointing down)
  Vector3d a_g_O_;

  /* Subscribers */
  rclcpp::Subscription<DvlMsg>::SharedPtr dvl_sub_;
  message_filters::Subscriber<ImuMsg> imu_sub_;;
  message_filters::Subscriber<DepthMsg> depth_sub_;

  using ApproximateTimeT =
    message_filters::sync_policies::ApproximateTime<ImuMsg, DepthMsg>;
  using SynchronizerT = message_filters::Synchronizer<ApproximateTimeT>;
  std::shared_ptr<SynchronizerT> synchronizer_;

  /* Publishers */
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;

  /* TF2 */
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr static_tf_timer_;

  SE3d body_T_dvl_;
  Vector3d r_body_depth_B_;
  // DVL adjoint matrix
  Matrix<double, 6, 6> Ad_dvl_T_body_;
  // DVL measurement Jacobian
  Matrix<double, 3, 6> F_dvl_;
  bool received_static_transforms_ = false;
};
