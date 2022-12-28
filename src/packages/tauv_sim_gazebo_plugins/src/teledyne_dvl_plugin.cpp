#include <boost/bind.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>

#include <tauv_sim_gazebo_plugins/teledyne_dvl_plugin.h>
#include <tauv_msgs/TeledyneDvlData.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::TeledyneDvlPlugin)

namespace gazebo {
void TeledyneDvlPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  this->model = model;
  GZ_ASSERT(sdf->HasElement("linkName"), "SDF missing linkName.");
  std::string linkName = sdf->Get<std::string>("linkName");
  this->link = this->model->GetLink(linkName);
  
  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&TeledyneDvlPlugin::OnUpdate, this, _1));
  
  this->angleOffset = this->link->RelativePose().Rot();

  if(!ros::isInitialized()) {
    throw "Attempted to create TeledyneDvlPlugin without initializing ROS";
  }
  this->rosNode.reset(new ros::NodeHandle("sim_teledyne_dvl_publisher"));
  this->rosPub = this->rosNode->advertise<tauv_msgs::TeledyneDvlData>("/vehicle/teledyne_dvl/data",10);
}

void TeledyneDvlPlugin::OnUpdate(const common::UpdateInfo &info)
{
  ignition::math::Vector3 velocity = this->angleOffset * this->model->RelativeLinearVel();
  tauv_msgs::TeledyneDvlData msg;
  geometry_msgs::Vector3 vec;
  //convert to our NED
  vec.x = -velocity.X();
  vec.y = -velocity.Y();
  vec.z = velocity.Z();
  msg.velocity = vec;
  this->rosPub.publish(msg);
}
}