#include <ros/ros.h>
#include "state_estimator.h"
#include <array>
#include <numeric>
#include <Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

StateEstimator::StateEstimator(ros::NodeHandle& n) : n(n)
{
  this->load_config();

  this->ekf = Ekf();
  this->ekf.set_dvl_offset(this->dvl_offset);
  this->ekf.set_process_covariance(this->process_covariance);

  this->imu_sub = n.subscribe("imu", 100, &StateEstimator::handle_imu, this);
  this->dvl_sub = n.subscribe("dvl", 100, &StateEstimator::handle_dvl, this);
  this->depth_sub = n.subscribe("depth", 100, &StateEstimator::handle_depth, this);

  this->pose_pub = n.advertise<tauv_msgs::Pose>("pose", 100);
  this->odom_pub = n.advertise<nav_msgs::Odometry>("odom", 100);

  this->timer = n.createTimer(this->dt, &StateEstimator::update, this);
}

void StateEstimator::load_config()
{
  int frequency;
  this->n.getParam("frequency", frequency);
  this->dt = ros::Duration(1.0 / double(frequency));

  double horizon_delay;
  this->n.getParam("horizon_delay", horizon_delay);
  this->horizon_delay = ros::Duration(horizon_delay);

  std::vector<double> dvl_offset;
  this->n.getParam("dvl_offset", dvl_offset);
  this->dvl_offset = Eigen::Map<Eigen::Vector3d>(dvl_offset.data());

  std::vector<double> process_covariance;
  this->n.getParam("process_covariance", process_covariance);
  this->process_covariance = Eigen::Map<Eigen::Matrix<double, 15, 1>>(process_covariance.data());

  std::vector<double> imu_covariance;
  this->n.getParam("imu_covariance", imu_covariance);
  this->imu_covariance = Eigen::Map<Eigen::Matrix<double, 9, 1>>(imu_covariance.data());

  std::vector<double> dvl_covariance;
  this->n.getParam("dvl_covariance", dvl_covariance);
  this->dvl_covariance = Eigen::Map<Eigen::Vector3d>(dvl_covariance.data());

  std::vector<double> depth_covariance;
  this->n.getParam("depth_covariance", depth_covariance);
  assert(depth_covariance.size() == 1);
  this->depth_covariance = depth_covariance.front(); 
}

Eigen::Quaterniond rpy_to_quat(Eigen::Vector3d &rpy)
{
    Eigen::AngleAxisd roll_angle(rpy.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_angle(rpy.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_angle(rpy.z(), Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yaw_angle * pitch_angle * roll_angle;
    return q;
}

void StateEstimator::update(const ros::TimerEvent& e)
{
  ros::Time current_time = ros::Time::now();
  ros::Time horizon_time = current_time - this->horizon_delay;

  while (!this->msg_queue.empty()) {
    StateEstimator::SensorMsg msg = this->msg_queue.top(); 
    this->msg_queue.pop();

    if (msg.time < horizon_time) {
      if (msg.is_imu()) {
          this->apply_imu(msg.as_imu());
      } else if (msg.is_dvl()) {
          this->apply_dvl(msg.as_dvl());
      } else if (msg.is_depth()) {
          this->apply_depth(msg.as_depth());
      }
    } else {
      this->msg_queue.push(msg);
      break;
    }
  }

  Eigen::Vector3d position, velocity, acceleration, orientation, angular_velocity;
  this->ekf.get_state(horizon_time.toSec(), position, velocity, acceleration, orientation, angular_velocity);

  tauv_msgs::Pose msg;
  msg.header.stamp = current_time;
  msg.position.x = position.x();
  msg.position.y = position.y();
  msg.position.z = position.z();
  msg.velocity.x = velocity.x();
  msg.velocity.y = velocity.y();
  msg.velocity.z = velocity.z();
  msg.acceleration.x = acceleration.x();
  msg.acceleration.y = acceleration.y();
  msg.acceleration.z = acceleration.z();
  msg.orientation.x = orientation.x();
  msg.orientation.y = orientation.y();
  msg.orientation.z = orientation.z();
  msg.angular_velocity.x = angular_velocity.x();
  msg.angular_velocity.y = angular_velocity.y();
  msg.angular_velocity.z = angular_velocity.z();
  this->pose_pub.publish(msg);


  Eigen::Quaterniond orientation_quat = rpy_to_quat(orientation);
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = current_time;
  odom_msg.header.frame_id = "odom"; 
  odom_msg.child_frame_id = "vehicle";
  odom_msg.pose.pose.position.x = position.x();
  odom_msg.pose.pose.position.y = position.y();
  odom_msg.pose.pose.position.z = position.z();
  odom_msg.pose.pose.orientation.x = orientation_quat.x();
  odom_msg.pose.pose.orientation.y = orientation_quat.y();
  odom_msg.pose.pose.orientation.z = orientation_quat.z();
  odom_msg.pose.pose.orientation.w = orientation_quat.w();
  odom_msg.twist.twist.linear.x = velocity.x();
  odom_msg.twist.twist.linear.y = velocity.y();
  odom_msg.twist.twist.linear.z = velocity.z();
  odom_msg.twist.twist.angular.x = angular_velocity.x();
  odom_msg.twist.twist.angular.y = angular_velocity.y();
  odom_msg.twist.twist.angular.z = angular_velocity.z();
  this->odom_pub.publish(odom_msg);

  tf::Transform odom_tf;
  odom_tf.setOrigin(tf::Vector3(position.x(), position.y(), position.z()));
  tf::Quaternion odom_quat;
  odom_quat.setRPY(orientation.x(), orientation.y(), orientation.z());
  odom_tf.setRotation(odom_quat);
  this->odom_tf_broadcaster.sendTransform(tf::StampedTransform(odom_tf, current_time, "odom", "vehicle"));

  this->last_horizon_time = horizon_time;
}

void StateEstimator::apply_imu(const tauv_msgs::XsensImuData::ConstPtr &msg)
{
  double time = msg->header.stamp.toSec();
  Eigen::Vector3d orientation { msg->orientation.x, msg->orientation.y, msg->orientation.z };

  Eigen::Vector3d free_acceleration { msg->free_acceleration.x, msg->free_acceleration.y, msg->free_acceleration.z };
  Eigen::Vector3d rate_of_turn { msg->rate_of_turn.x, msg->rate_of_turn.y, msg->rate_of_turn.z };

  Eigen::Quaterniond orientation_quat = rpy_to_quat(orientation);
  Eigen::Vector3d linear_acceleration = orientation_quat.matrix().inverse() * free_acceleration;

  this->ekf.handle_imu_measurement(time, orientation, rate_of_turn, linear_acceleration, this->imu_covariance);
}

void StateEstimator::apply_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr &msg)
{
  double time = msg->header.stamp.toSec();

  if (!msg->is_hr_velocity_valid) return;

  Eigen::Vector3d velocity { msg->hr_velocity.x, msg->hr_velocity.y, msg->hr_velocity.z }; 

  double avg_covariance = 0.0;
  std::accumulate(msg->beam_standard_deviations.begin(), msg->beam_standard_deviations.end(), avg_covariance);
  avg_covariance /= 4.0;

  Eigen::Vector3d covariance { avg_covariance, avg_covariance, avg_covariance };
  covariance = covariance.cwiseProduct(this->dvl_covariance);

  this->ekf.handle_dvl_measurement(time, velocity, covariance);
}

void StateEstimator::apply_depth(const tauv_msgs::FluidDepth::ConstPtr &msg)
{
  double time = msg->header.stamp.toSec();

  this->ekf.handle_depth_measurement(time, msg->depth, this->depth_covariance); 
}

void StateEstimator::handle_imu(const tauv_msgs::XsensImuData::ConstPtr& msg)
{
  StateEstimator::SensorMsg sensor_msg(msg);
  this->msg_queue.push(sensor_msg);
}

void StateEstimator::handle_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr& msg)
{
  StateEstimator::SensorMsg sensor_msg(msg);
  this->msg_queue.push(sensor_msg);
}

void StateEstimator::handle_depth(const tauv_msgs::FluidDepth::ConstPtr& msg)
{
  StateEstimator::SensorMsg sensor_msg(msg);
  this->msg_queue.push(sensor_msg);
}
