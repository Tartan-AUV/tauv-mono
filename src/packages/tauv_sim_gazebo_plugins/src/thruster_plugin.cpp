#include <boost/bind.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <tauv_sim_gazebo_plugins/thruster_plugin.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::ThrusterPlugin)

namespace gazebo {
void ThrusterPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  this->model = model;
  GZ_ASSERT(sdf->HasElement("linkName"), "SDF missing linkName.");
  std::string linkName = sdf->Get<std::string>("linkName");
  this->link = this->model->GetLink(linkName);

  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&ThrusterPlugin::OnUpdate, this, _1));
}

void ThrusterPlugin::OnUpdate(const common::UpdateInfo &info)
{
  ignition::math::Vector3d force(100, 0, 0);
  this->link->AddRelativeForce(force);
}
}