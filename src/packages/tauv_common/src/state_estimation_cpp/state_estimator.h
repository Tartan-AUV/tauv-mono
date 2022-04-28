#include <array>
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

#pragma once

class StateEstimator {
  public:
    StateEstimator(ros::NodeHandle& n);

    void update(const ros::TimerEvent& e);

    void handle_imu(const tauv_msgs::XsensImuData::ConstPtr& msg);
    void handle_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr& msg);
    void handle_depth(const tauv_msgs::FluidDepth::ConstPtr& msg);

    class SensorMsg;
    class Checkpoint;

  private:
    ros::NodeHandle& n;
    ros::Timer timer;

    tauv_alarms::AlarmClient alarm_client;

    ros::Subscriber imu_sub;
    ros::Subscriber dvl_sub;
    ros::Subscriber depth_sub;

    ros::Publisher pose_pub;
    ros::Publisher odom_pub;

    tf::TransformBroadcaster odom_tf_broadcaster;

    Ekf ekf;

    bool is_initialized;

    ros::Time last_evaluation_time;

    boost::circular_buffer<Checkpoint> checkpoints;

    boost::heap::priority_queue<SensorMsg, boost::heap::compare<std::greater_equal<SensorMsg>>> realtime_queue;
    boost::heap::priority_queue<SensorMsg, boost::heap::compare<std::greater_equal<SensorMsg>>> delayed_queue;

    ros::Duration dt;
    Eigen::Vector3d dvl_offset;
    Eigen::Matrix<double, 15, 1> process_covariance;
    Eigen::Matrix<double, 9, 1> imu_covariance;
    Eigen::Vector3d dvl_covariance;
    double depth_covariance;

    void load_config();

    void apply_imu(const tauv_msgs::XsensImuData::ConstPtr &msg);
    void apply_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr &msg);
    void apply_depth(const tauv_msgs::FluidDepth::ConstPtr &msg);
};

class StateEstimator::SensorMsg {
  public:
    enum Type { IMU, DVL, DEPTH };

    Type type;
    ros::Time stamp;
    boost::variant<tauv_msgs::XsensImuData::ConstPtr, tauv_msgs::TeledyneDvlData::ConstPtr, tauv_msgs::FluidDepth::ConstPtr> msg;

    SensorMsg(const tauv_msgs::XsensImuData::ConstPtr &raw_msg) : msg(raw_msg) {
      this->type = Type::IMU;  
      this->stamp = raw_msg->header.stamp;
    };
    SensorMsg(const tauv_msgs::TeledyneDvlData::ConstPtr &raw_msg) : msg(raw_msg) {
      this->type = Type::DVL;  
      this->stamp = raw_msg->header.stamp;
    };
    SensorMsg(const tauv_msgs::FluidDepth::ConstPtr &raw_msg) : msg(raw_msg) {
      this->type = Type::DEPTH;  
      this->stamp = raw_msg->header.stamp;
    };
    
    bool is_imu() { return this->type == Type::IMU; };
    bool is_dvl() { return this->type == Type::DVL; };
    bool is_depth() { return this->type == Type::DEPTH; };

    const tauv_msgs::XsensImuData::ConstPtr& as_imu() {
      return boost::get<const tauv_msgs::XsensImuData::ConstPtr&>(this->msg);
    };
    const tauv_msgs::TeledyneDvlData::ConstPtr& as_dvl() {
      return boost::get<const tauv_msgs::TeledyneDvlData::ConstPtr&>(this->msg);
    };
    const tauv_msgs::FluidDepth::ConstPtr& as_depth() {
      return boost::get<const tauv_msgs::FluidDepth::ConstPtr&>(this->msg);
    };
};

bool operator < (const StateEstimator::SensorMsg &lhs, const StateEstimator::SensorMsg &rhs) 
{
  return lhs.stamp < rhs.stamp;
}

bool operator >= (const StateEstimator::SensorMsg &lhs, const StateEstimator::SensorMsg &rhs) 
{
  return lhs.stamp >= rhs.stamp;
}

class StateEstimator::Checkpoint {
  public:
    Checkpoint(const StateEstimator::SensorMsg &msg) : msg(msg)
    {
      this->stamp = msg.stamp;
    };

    ros::Time stamp;
    StateEstimator::SensorMsg msg;
    Eigen::Matrix<double, 15, 1> state;
    Eigen::Matrix<double, 15, 15> cov;
};
