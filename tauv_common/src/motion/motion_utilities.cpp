#include "motion_utilities.h"

#include <ros/exception.h>
#include <string>

#include <geometry_msgs/Twist.h>
#include "constructors/Twist.h"
#include "constructors/Vector3.h"

Mover::Mover(ros::NodeHandle &nh)
{
    std::string mover_topic = "/tartan_sub/mover";
    vel_pub_ = nh.advertise<geometry_msgs::Twist>(mover_topic, 1);
}

void Mover::dive(float duration, float speed)
{
    geometry_msgs::Twist msg = 
        Twist(Vector3(0, 0, speed), Vector3(0, 0, 0));
    publishDuration(msg, duration);
}

void Mover::forward(float duration, float speed)
{
    geometry_msgs::Twist msg = 
        Twist(Vector3(speed, 0, 0), Vector3(0, 0, 0));
    publishDuration(msg, duration);
}

void Mover::strafe(float duration, float speed)
{
    geometry_msgs::Twist msg = 
        Twist(Vector3(0, speed, 0), Vector3(0, 0, 0));
    publishDuration(msg, duration);
}

void Mover::turn(float duration, float speed)
{
    geometry_msgs::Twist msg = 
        Twist(Vector3(0, 0, 0), Vector3(0, 0, speed));
    publishDuration(msg, duration);
}

void Mover::commonEnd()
{
    if (ros::ok())
        publishMessage(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)));
}

void Mover::publishDuration(geometry_msgs::Twist vel, float duration)
{
    float end_time = (float)ros::Time::now().toSec() + duration;
    while (ros::Time::now().toSec() < end_time && ros::ok())
        publishMessage(vel);
    commonEnd();
}

void Mover::publishMessage(geometry_msgs::Twist vel)
{
    vel_pub_.publish(vel);
}
