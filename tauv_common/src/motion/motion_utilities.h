#ifndef MOTION_MOTION_UTILITIES_H
#define MOTION_MOTION_UTILITIES_H
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>

#include "units/units.h"

using namespace units::time;
using namespace units::velocity;

class Mover {
    public:
        Mover(ros::NodeHandle &nh);

        // move in z direction
        void dive(second_t duration, meters_per_second_t speed);

        // move in x direction
        void forward(second_t duration, meters_per_second_t speed);

        // move in y direction
        void strafe(second_t duration, meters_per_second_t speed);

        // angular turn in z direction
        void turn(second_t duration, meters_per_second_t speed);

    private:
        // stop sending messages
        void commonEnd();

        // publish message for duration
        void publishDuration(geometry_msgs::Twist vel, second_t duration);

        // publish message
        void publishMessage(geometry_msgs::Twist vel);

        ros::Publisher vel_pub_;
};

#endif  // MOTION_UTILITIES.H
