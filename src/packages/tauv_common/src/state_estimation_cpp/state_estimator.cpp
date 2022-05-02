#include <boost/circular_buffer.hpp>
#include <boost/heap/priority_queue.hpp>
#include <boost/variant.hpp>
#include <eigen3/Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <numeric>
#include <ros/ros.h>
#include <tauv_msgs/FluidDepth.h>
#include <tauv_msgs/Pose.h>
#include <tauv_msgs/TeledyneDvlData.h>
#include <tauv_msgs/XsensImuData.h>
#include <tf/transform_broadcaster.h>
#include <tauv_alarms/alarms.h>
#include <tauv_alarms/alarm_client.h>
#include "ekf.h"
#include "state_estimator.h"

StateEstimator::StateEstimator(ros::NodeHandle& n) : n(n), alarm_client(n), transform_client(n)
{
  this->load_config();

  this->is_initialized = false;

  this->ekf = Ekf();
  this->ekf.set_dvl_offset(this->dvl_offset);
  this->ekf.set_process_covariance(this->process_covariance);

  this->imu_sub = n.subscribe("imu", 100, &StateEstimator::handle_imu, this);
  this->dvl_sub = n.subscribe("dvl", 100, &StateEstimator::handle_dvl, this);
  this->depth_sub = n.subscribe("depth", 100, &StateEstimator::handle_depth, this);

  this->pose_pub = n.advertise<tauv_msgs::Pose>("pose", 100);
  this->odom_pub = n.advertise<nav_msgs::Odometry>("odom", 100);

  this->timer = n.createTimer(this->dt, &StateEstimator::update, this);

  this->checkpoints = boost::circular_buffer<StateEstimator::Checkpoint>(400);

  this->alarm_client.clear(tauv_alarms::AlarmType::STATE_ESTIMATION_NOT_INITIALIZED, "State estimation initialized.");
}

void StateEstimator::load_config()
{
  int frequency;
  this->n.getParam("frequency", frequency);
  this->dt = ros::Duration(1.0 / double(frequency));

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
    Eigen::AngleAxisd roll_angle(-rpy.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_angle(-rpy.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_angle(-rpy.z(), Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yaw_angle * pitch_angle * roll_angle;

    return q;
}

void StateEstimator::update(const ros::TimerEvent& e)
{
  if (!this->is_initialized) return;

  ros::Time current_time = ros::Time::now() - ros::Duration(0.05);

  ROS_INFO("delayed: %ld, realtime: %ld", this->delayed_queue.size(), this->realtime_queue.size());

  if (!this->delayed_queue.empty() && this->checkpoints.full()) {
    SensorMsg msg = this->delayed_queue.top();
    this->delayed_queue.pop();

    boost::circular_buffer<Checkpoint>::iterator it = this->checkpoints.begin();
    Checkpoint previous_checkpoint(it->msg);
    while (it != this->checkpoints.end() && it->stamp <= msg.stamp) {
      previous_checkpoint = *it;
      ++it;
    }
    if (previous_checkpoint.stamp > msg.stamp) {
      ROS_WARN("previous checkpoint stamp later than msg stamp");
    } else {
        // Revert state to previous checkpoint
        // Then copy all the messages from the rest of the checkpoint buffer to a queue
        // Squash all those messages and write the resulting checkpoints into the checkpoint buffer starting at the previous_checkpoint

        this->ekf.set_state(previous_checkpoint.state);
        this->ekf.set_cov(previous_checkpoint.cov);
        this->ekf.set_time(previous_checkpoint.stamp.toSec());


        boost::circular_buffer<Checkpoint>::iterator copy_it = it;
        boost::heap::priority_queue<SensorMsg, boost::heap::compare<std::greater_equal<SensorMsg>>> msgs;
        msgs.push(msg);
        while (copy_it != this->checkpoints.end()) {
          msgs.push(copy_it->msg);
          ++copy_it;
        }

        this->checkpoints.erase_end(std::distance(it, this->checkpoints.end()));

        while (!this->delayed_queue.empty()) {
          SensorMsg msg = this->delayed_queue.top();
          this->delayed_queue.pop();
          msgs.push(msg);
        }

        while (!msgs.empty()) {
          StateEstimator::SensorMsg msg = msgs.top();
          msgs.pop();

          if (msg.is_imu()) {
              this->apply_imu(msg.as_imu());
          } else if (msg.is_dvl()) {
              this->apply_dvl(msg.as_dvl());
          } else if (msg.is_depth()) {
              this->apply_depth(msg.as_depth());
          }

          Checkpoint c(msg);
          this->ekf.get_state(c.state);
          this->ekf.get_cov(c.cov);
          this->checkpoints.push_back(c);
        }
    }
  }

  while (!this->realtime_queue.empty()) {
    StateEstimator::SensorMsg msg = this->realtime_queue.top();
    this->realtime_queue.pop();

    if (msg.is_imu()) {
        this->apply_imu(msg.as_imu());
    } else if (msg.is_dvl()) {
        this->apply_dvl(msg.as_dvl());
    } else if (msg.is_depth()) {
        this->apply_depth(msg.as_depth());
    }

    Checkpoint c(msg);
    this->ekf.get_state(c.state);
    this->ekf.get_cov(c.cov);
    this->checkpoints.push_back(c);
  }

  this->last_evaluation_time = current_time;

  Eigen::Vector3d position, velocity, acceleration, orientation, angular_velocity;
  this->ekf.get_state_fields(current_time.toSec(), position, velocity, acceleration, orientation, angular_velocity);

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
  odom_msg.header.frame_id = "odom_ned";
  odom_msg.child_frame_id = "vehicle_ned";
  odom_msg.pose.pose.position.x = position.x();
  odom_msg.pose.pose.position.y = position.y();
  odom_msg.pose.pose.position.z = position.z();
  odom_msg.pose.pose.orientation.x = orientation_quat.x();
  odom_msg.pose.pose.orientation.y = orientation_quat.y();
  odom_msg.pose.pose.orientation.z = orientation_quat.z();
  odom_msg.pose.pose.orientation.w = -orientation_quat.w();
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
  this->odom_tf_broadcaster.sendTransform(tf::StampedTransform(odom_tf, current_time, "odom_ned", "vehicle_ned"));
}

void StateEstimator::apply_imu(const tauv_msgs::XsensImuData::ConstPtr &msg)
{
  double time = msg->header.stamp.toSec();
  Eigen::Vector3d orientation { -msg->orientation.x, msg->orientation.y, -msg->orientation.z };

  Eigen::Vector3d linear_acceleration { -msg->linear_acceleration.x, msg->linear_acceleration.y, -msg->linear_acceleration.z };
  Eigen::Vector3d rate_of_turn { -msg->rate_of_turn.x, msg->rate_of_turn.y, -msg->rate_of_turn.z };

  Eigen::Quaterniond orientation_quat = rpy_to_quat(orientation);
  Eigen::Vector3d gravity { 0.0, 0.0, 9.806 };
  Eigen::Vector3d body_gravity = orientation_quat.inverse().toRotationMatrix() * gravity;
  body_gravity.z() = -1 * body_gravity.z();
  Eigen::Vector3d free_acceleration = linear_acceleration - body_gravity;

  this->ekf.handle_imu_measurement(time, orientation, rate_of_turn, free_acceleration, this->imu_covariance);
}

void StateEstimator::apply_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr &msg)
{
  double time = msg->header.stamp.toSec();

  if (!msg->is_hr_velocity_valid) return;

  Eigen::Vector3d velocity { msg->hr_velocity.y, msg->hr_velocity.x, -msg->hr_velocity.z }; 

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
  if (!this->is_initialized) {
    this->is_initialized = true;
    this->last_evaluation_time = msg->header.stamp - ros::Duration(0.05);
  }
  SensorMsg sensor_msg(msg);
  if (msg->header.stamp < this->last_evaluation_time && this->checkpoints.full()) {
    this->delayed_queue.push(sensor_msg);
  } else {
    this->realtime_queue.push(sensor_msg);
  }
}

void StateEstimator::handle_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr& msg)
{
  if (!this->is_initialized) {
    this->is_initialized = true;
    this->last_evaluation_time = msg->header.stamp - ros::Duration(0.05);
  }

  // if (!msg->is_hr_velocity_valid) return;
  SensorMsg sensor_msg(msg);
  if (msg->header.stamp < this->last_evaluation_time && this->checkpoints.full()) {
    this->delayed_queue.push(sensor_msg);
  } else {
    this->realtime_queue.push(sensor_msg);
  }
}

void StateEstimator::handle_depth(const tauv_msgs::FluidDepth::ConstPtr& msg)
{
  if (!this->is_initialized) {
    this->is_initialized = true;
    this->last_evaluation_time = msg->header.stamp - ros::Duration(0.05);
  }
  SensorMsg sensor_msg(msg);
  if (msg->header.stamp < this->last_evaluation_time && this->checkpoints.full()) {
    this->delayed_queue.push(sensor_msg);
  } else {
    this->realtime_queue.push(sensor_msg);
  }
}
