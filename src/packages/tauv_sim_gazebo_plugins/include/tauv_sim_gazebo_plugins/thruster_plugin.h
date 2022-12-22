
#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo
{
class ThrusterPlugin : public ModelPlugin
{
  public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf);
  void OnUpdate(const common::UpdateInfo &info);

  protected:
  physics::ModelPtr model;
  physics::LinkPtr link;
  event::ConnectionPtr updateConnection;
};
}