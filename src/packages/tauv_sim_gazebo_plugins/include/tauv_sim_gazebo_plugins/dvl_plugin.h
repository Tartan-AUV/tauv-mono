
#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>

namespace gazebo
{
class DvlPlugin : public ModelPlugin
{
  public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf);
  void OnUpdate(const common::UpdateInfo &info);

  protected:
  physics::ModelPtr model;
  physics::LinkPtr link;
  event::ConnectionPtr updateConnection;
  std::unique_ptr<ros::NodeHandle> rosNode;
  ros::Publisher rosPub;
};
}