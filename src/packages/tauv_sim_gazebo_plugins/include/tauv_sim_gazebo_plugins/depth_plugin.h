#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>

namespace gazebo {
class DepthPlugin : public ModelPlugin
{
    public:
    void Load(physics::ModelPtr model, sdf::ElementPtr sdf);
    void OnUpdate(const common::UpdateInfo &info);

    private:
    event::ConnectionPtr updateConnection;

    boost::scoped_ptr<ros::NodeHandle> rosNode;

    physics::ModelPtr model;
    physics::LinkPtr link;

    ros::Publisher pubDepth;

    common::Time publishPeriod;
    common::Time lastPublishTime;
};
}
