#include <boost/bind.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>

#include <tauv_sim_gazebo_plugins/dvl_plugin.h>
#include <tauv_msgs/TeledyneDvlData.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::DvlPlugin)

namespace gazebo {
void DvlPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  this->model = model;
  GZ_ASSERT(sdf->HasElement("linkName"), "SDF missing linkName.");
  std::string linkName = sdf->Get<std::string>("linkName");
  this->link = this->model->GetLink(linkName);
  
  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&DvlPlugin::OnUpdate, this, _1));

  if(!ros::isInitialized()) {
    throw "Attempted to create DvlPlugin without initializing ROS";
  }
  this->rosNode.reset(new ros::NodeHandle("dvl_publisher"));
  this->rosPub = this->rosNode->advertise<tauv_msgs::TeledyneDvlData>("/vehicle/teledyne_dvl/data",10);
}

void DvlPlugin::OnUpdate(const common::UpdateInfo &info)
{
  ignition::math::Vector3 velocity = this->model->RelativeLinearVel();
  tauv_msgs::TeledyneDvlData msg;
  geometry_msgs::Vector3 vec;
  vec.x = velocity.X();
  vec.y = velocity.Y();
  vec.z = velocity.Z();
  msg.velocity = vec;
  this->rosPub.publish(msg);
}
}