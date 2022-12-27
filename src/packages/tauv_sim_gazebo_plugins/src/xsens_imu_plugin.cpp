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
geometry_msgs::Vector3 FromIgnitionVector3(ignition::math::Vector3<double> vec) {
  geometry_msgs::Vector3 res;
  res.x = vec.X();
  res.y = vec.Y();
  res.z = vec.Z();
  return res;
}

inline ignition::math::Vector3<double> IgnitionVector3(double x, double y, double z){
    ignition::math::Vector3<double> res;
    res.X(x);
    res.Y(y);
    res.Z(z);
    return res;
}

inline ignition::math::Vector3<double> AddGravity(ignition::math::Vector3d relativeAccel, ignition::math::Quaternion<double> rot) {
    //make sure this points the right way
    return relativeAccel + rot.RotateVector(IgnitionVector3(0,0,-9.81));
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
 // this->angleOffset = this->link->RelativePose().Rot();
}

void XsensImuPlugin::OnUpdate(const common::UpdateInfo &info)
{
  tauv_msgs::XsensImuData msg;
  ignition::math::Quaternion rot = this->model->RelativePose().Rot();
  ignition::math::Vector3 freeAccel = this->model->RelativeLinearAccel(); 
  msg.orientation = FromIgnitionVector3(rot.Euler());
  msg.free_acceleration = FromIgnitionVector3(freeAccel);
  msg.rate_of_turn = FromIgnitionVector3(this->model->RelativeAngularVel());
  msg.linear_acceleration = FromIgnitionVector3(AddGravity(freeAccel, rot));
  //TODO make sure these are the right units and it's the right frame
  //std::cout << this->angleOffset.Euler() << std::endl;
  this->rosPub.publish(msg);
}
}