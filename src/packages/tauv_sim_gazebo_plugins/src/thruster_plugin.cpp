#include <boost/bind.hpp>
#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>
#include <std_msgs/Float64.h>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <tauv_sim_gazebo_plugins/thruster_plugin.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::ThrusterPlugin)

namespace gazebo {
void ThrusterPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  if (!ros::isInitialized())
  {
    gzerr << "ROS not initialized." << std::endl;
    return;
  }

  this->rosNode.reset(new ros::NodeHandle("thruster_0_plugin"));

  this->model = model;

  GZ_ASSERT(sdf->HasElement("linkName"), "SDF missing linkName.");
  std::string linkName = sdf->Get<std::string>("linkName");
  this->link = this->model->GetLink(linkName);

  GZ_ASSERT(sdf->HasElement("thrusterID"), "SDF missing thrusterID.");
  this->thrusterID = sdf->Get<int>("thrusterID");

  std::string targetThrustTopic = "/kingfisher/thrusters/" + std::to_string(this->thrusterID) + "/target_thrust";
  std::string thrustTopic = "/kingfisher/thrusters/" + std::to_string(this->thrusterID) + "/thrust";

  this->subTargetThrust = this->rosNode->subscribe<std_msgs::Float64>(
    targetThrustTopic,
    10,
    boost::bind(&ThrusterPlugin::HandleTargetThrust, this, _1)
  );

  this->pubThrust = this->rosNode->advertise<std_msgs::Float64>(
    thrustTopic, 10
  );

  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&ThrusterPlugin::OnUpdate, this, _1)
  );
}

void ThrusterPlugin::OnUpdate(const common::UpdateInfo& info)
{
  ignition::math::Vector3d force(this->targetThrust, 0, 0);
  this->link->AddRelativeForce(force);
}

void ThrusterPlugin::HandleTargetThrust(const std_msgs::Float64::ConstPtr& msg)
{
  this->targetThrust = msg->data;
}
}
