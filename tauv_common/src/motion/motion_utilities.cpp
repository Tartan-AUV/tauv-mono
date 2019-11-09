#include "motion_utilities.h"

#include <ros/exception.h>
#include <string>

#include <geometry_msgs/Twist.h>
#include "constructors/Twist.h"
#include "constructors/Vector3.h"

<<<<<<< HEAD
Mover::Mover(ros::NodeHandle &nh)
{
=======
Mover::Mover(ros::NodeHandle &nh) {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    std::string mover_topic = "/tartan_sub/mover";
    vel_pub_ = nh.advertise<geometry_msgs::Twist>(mover_topic, 1);
}

<<<<<<< HEAD
void Mover::dive(float duration, float speed)
{
=======
void Mover::dive(float duration, float speed) {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    geometry_msgs::Twist msg = 
        Twist(Vector3(0, 0, speed), Vector3(0, 0, 0));
    publishDuration(msg, duration);
}

<<<<<<< HEAD
void Mover::forward(float duration, float speed)
{
=======
void Mover::forward(float duration, float speed) {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    geometry_msgs::Twist msg = 
        Twist(Vector3(speed, 0, 0), Vector3(0, 0, 0));
    publishDuration(msg, duration);
}

<<<<<<< HEAD
void Mover::strafe(float duration, float speed)
{
=======
void Mover::strafe(float duration, float speed) {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    geometry_msgs::Twist msg = 
        Twist(Vector3(0, speed, 0), Vector3(0, 0, 0));
    publishDuration(msg, duration);
}

<<<<<<< HEAD
void Mover::turn(float duration, float speed)
{
=======
void Mover::turn(float duration, float speed) {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    geometry_msgs::Twist msg = 
        Twist(Vector3(0, 0, 0), Vector3(0, 0, speed));
    publishDuration(msg, duration);
}

<<<<<<< HEAD
void Mover::commonEnd()
{
=======
void Mover::commonEnd() {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    if (ros::ok())
        publishMessage(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)));
}

<<<<<<< HEAD
void Mover::publishDuration(geometry_msgs::Twist vel, float duration)
{
=======
void Mover::publishDuration(geometry_msgs::Twist vel, float duration) {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    float end_time = (float)ros::Time::now().toSec() + duration;
    while (ros::Time::now().toSec() < end_time && ros::ok())
        publishMessage(vel);
    commonEnd();
}

<<<<<<< HEAD
void Mover::publishMessage(geometry_msgs::Twist vel)
{
=======
void Mover::publishMessage(geometry_msgs::Twist vel) {
>>>>>>> f9e38572f42b0d362a9bc601b80821244139e918
    vel_pub_.publish(vel);
}
