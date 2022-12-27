
#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <ros/ros.h>

namespace gazebo
{
class XsensImuPlugin : public ModelPlugin
{
  public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf);
  void OnUpdate(const common::UpdateInfo &info);
  inline ignition::math::Vector3<double> ToRobotFrame(ignition::math::Vector3<double> original) {
    return this->angleOffset.RotateVectorReverse(original);
  }
  
  protected:
  physics::ModelPtr model;
  physics::LinkPtr link;
  event::ConnectionPtr updateConnection;
  std::unique_ptr<ros::NodeHandle> rosNode;
  ros::Publisher rosPub;
  ignition::math::Quaterniond angleOffset;
};
}