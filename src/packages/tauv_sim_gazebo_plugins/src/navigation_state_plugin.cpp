#include <ros/ros.h>
#include <boost/scoped_ptr.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/math/Matrix4.hh>

#include <tauv_msgs/NavigationState.h>

#include <tauv_sim_gazebo_plugins/navigation_state_plugin.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::NavigationStatePlugin)

namespace gazebo {
void NavigationStatePlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
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

  GZ_ASSERT(sdf->HasElement("navigationStateTopic"), "SDF missing navigationStateTopic.");
  std::string navigationStateTopic = sdf->Get<std::string>("navigationStateTopic");

  this->rosNode.reset(new ros::NodeHandle(nodeName));

  this->pubNavState = this->rosNode->advertise<tauv_msgs::NavigationState>(
    navigationStateTopic, 10
  );

  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&NavigationStatePlugin::OnUpdate, this, _1)
  );
}

void NavigationStatePlugin::OnUpdate(const common::UpdateInfo& info)
{
  if (info.simTime - this->lastPublishTime < this->publishPeriod) {
    return;
  }

  ignition::math::Pose3d linkPose = this->link->WorldPose();
  ignition::math::Vector3d linkLinVelENU = this->link->WorldLinearVel();
  ignition::math::Vector3d linkLinVel(
    linkLinVelENU.X(),
    -linkLinVelENU.Y(),
    -linkLinVelENU.Z()
  );
  ignition::math::Vector3d linkAngVelENU = this->link->WorldAngularVel();
  ignition::math::Vector3d linkAngVel(
    linkAngVelENU.X(),
    -linkAngVelENU.Y(),
    -linkAngVelENU.Z()
  );

  ignition::math::Quaterniond rot(
    linkPose.Rot().W(),
    linkPose.Rot().X(),
    -linkPose.Rot().Y(),
    -linkPose.Rot().Z()
  );

  ignition::math::Vector3d linkLinAccelENU = this->link->WorldLinearAccel();
  ignition::math::Vector3d linkLinAccel(
    linkLinAccelENU.X(),
    -linkLinAccelENU.Y(),
    -linkLinAccelENU.Z()
  );

  ignition::math::Vector3d linVel = rot.Inverse() * linkLinVel;
  ignition::math::Vector3d linAccel = rot.Inverse() * linkLinAccel;

  ignition::math::Matrix3d angVelToEulerRate(
    1, sin(rot.Roll()) * tan(rot.Pitch()), cos(rot.Roll()) * tan(rot.Pitch()),
    0, cos(rot.Roll()), -sin(rot.Roll()),
    0, sin(rot.Roll()) / cos(rot.Pitch()), cos(rot.Roll()) / cos(rot.Pitch())
  );

  ignition::math::Vector3d eulerRate = angVelToEulerRate * (rot.Inverse() * linkAngVel);

  tauv_msgs::NavigationState navStateMsg;
  navStateMsg.position.x = linkPose.Pos().X();
  navStateMsg.position.y = -linkPose.Pos().Y();
  navStateMsg.position.z = -linkPose.Pos().Z();

  navStateMsg.orientation.x = rot.Roll();
  navStateMsg.orientation.y = rot.Pitch();
  navStateMsg.orientation.z = rot.Yaw();

  navStateMsg.velocity.x = linVel.X();
  navStateMsg.velocity.y = linVel.Y();
  navStateMsg.velocity.z = linVel.Z();

  navStateMsg.angular_velocity.x = eulerRate.X();
  navStateMsg.angular_velocity.y = eulerRate.Y();
  navStateMsg.angular_velocity.z = eulerRate.Z();

  navStateMsg.acceleration.x = linAccel.X();
  navStateMsg.acceleration.y = linAccel.Y();
  navStateMsg.acceleration.z = linAccel.Z();

  this->pubNavState.publish(navStateMsg);

  this->lastPublishTime = info.simTime;
}
}
