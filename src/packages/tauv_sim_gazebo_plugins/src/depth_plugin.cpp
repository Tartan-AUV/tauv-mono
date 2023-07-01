#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>

#include <tauv_msgs/FluidDepth.h>

#include <tauv_sim_gazebo_plugins/depth_plugin.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::DepthPlugin)

namespace gazebo {
void DepthPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  if (!ros::isInitialized())
  {
    gzerr << "ROS not initialized." << std::endl;
    return;
  }

  this->model = model;

  GZ_ASSERT(sdf->HasElement("linkName"), "SDF missing linkName.");
  std::string linkName = sdf->Get<std::string>("linkName");
  this->link = this->model->GetLink(linkName);

  GZ_ASSERT(sdf->HasElement("publishRate"), "SDF missing publishRate.");
  this->publishPeriod = 1.0 / (double)sdf->Get<int>("publishRate");

  GZ_ASSERT(sdf->HasElement("nodeName"), "SDF missing nodeName.");
  std::string nodeName = sdf->Get<std::string>("nodeName");

  GZ_ASSERT(sdf->HasElement("depthTopic"), "SDF missing depthTopic.");
  std::string depthTopic = sdf->Get<std::string>("depthTopic");

  this->rosNode.reset(new ros::NodeHandle(nodeName));

  this->pubDepth = this->rosNode->advertise<tauv_msgs::FluidDepth>(depthTopic, 10);

  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&DepthPlugin::OnUpdate, this, _1)
  );
}

void DepthPlugin::OnUpdate(const common::UpdateInfo& info)
{
  if (info.simTime - this->lastPublishTime < this->publishPeriod) {
    return;
  }

  ignition::math::Pose3d linkPose = this->link->WorldPose();

  ros::Time current_time = ros::Time::now();

  tauv_msgs::FluidDepth depth_msg;
  depth_msg.header.stamp = current_time;
  depth_msg.depth = -linkPose.Pos().Z();

  this->pubDepth.publish(depth_msg);
}
}
