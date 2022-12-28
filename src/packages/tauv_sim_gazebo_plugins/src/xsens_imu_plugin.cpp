#include <boost/bind.hpp>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>

#include <tauv_sim_gazebo_plugins/xsens_imu_plugin.h>
#include <tauv_msgs/XsensImuData.h>

GZ_REGISTER_MODEL_PLUGIN(gazebo::XsensImuPlugin)

namespace gazebo {
//returns NED
geometry_msgs::Vector3 FromIgnitionVector3(ignition::math::Vector3<double> vec) {
  geometry_msgs::Vector3 res;
  res.x = -vec.X();
  res.y = -vec.Y();
  res.z = vec.Z();
  return res;
}

inline ignition::math::Vector3<double> AddGravity(ignition::math::Vector3d relativeAccel, ignition::math::Quaternion<double> rot) {
    //make sure this points the right way
    return relativeAccel + rot.RotateVector(ignition::math::Vector3d(0,0,-9.81));
}

void XsensImuPlugin::Load(physics::ModelPtr model, sdf::ElementPtr sdf)
{
  this->model = model;
  GZ_ASSERT(sdf->HasElement("linkName"), "SDF missing linkName.");
  std::string linkName = sdf->Get<std::string>("linkName");

  this->link = this->model->GetLink(linkName);
  
  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    boost::bind(&XsensImuPlugin::OnUpdate, this, _1));

  if(!ros::isInitialized()) {
    throw "Attempted to create XsensImuPlugin without initializing ROS";
  }
  this->rosNode.reset(new ros::NodeHandle("sim_xsens_imu_publisher"));
  this->rosPub = this->rosNode->advertise<tauv_msgs::XsensImuData>("/vehicle/xsens_imu/data",10);
  
  // Offset for the initial angle of the IMU linkage: 
  // this may not always be the desired behavior, maybe add a param for it?
  //anyway, this first rotates the IMU relative to the robot, and offsets the readings to
  //account for the robot start angle (assuming the IMU considers this to be 0)
  // this->angleOffset = this->model->WorldPose().Rot() * this->link->RelativePose().Rot();

  //this is for the offset that doesn't zero the IMU on start
  this->angleOffset = this->link->RelativePose().Rot();
}

void XsensImuPlugin::OnUpdate(const common::UpdateInfo &info)
{
  tauv_msgs::XsensImuData msg;
  ignition::math::Quaternion rot = this->model->RelativePose().Rot();
  ignition::math::Matrix3d angVelToEulerRate(
    1, sin(rot.Roll()) * tan(rot.Pitch()), cos(rot.Roll()) * tan(rot.Pitch()),
    0, cos(rot.Roll()), -sin(rot.Roll()),
    0, sin(rot.Roll()) / cos(rot.Pitch()), cos(rot.Roll()) / cos(rot.Pitch())
  );
  ignition::math::Vector3 freeAccel = this->model->RelativeLinearAccel();
  msg.orientation = FromIgnitionVector3((this->angleOffset.Inverse()*rot).Euler());
  msg.free_acceleration = FromIgnitionVector3(this->angleOffset.Inverse()*freeAccel);
  msg.rate_of_turn = FromIgnitionVector3(angVelToEulerRate * (this->angleOffset.Inverse()*this->model->RelativeAngularVel()));
  msg.linear_acceleration = FromIgnitionVector3(this->angleOffset.Inverse()*AddGravity(freeAccel, rot));
  this->rosPub.publish(msg);
}
}