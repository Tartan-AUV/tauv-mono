#ifndef MOTION_MOTION_UTILITIES_H
#define MOTION_MOTION_UTILITIES_H
#include <geometry_msgs/Twist.h>
#include <ros/ros.h>

#include "units/units.h"

class Mover {
    public:
        Mover(ros::NodeHandle &nh);

        // move in z direction
        void dive(float duration, float speed);

        // move in x direction
        void forward(float duration, float speed);

        // move in y direction
        void strafe(float duration, float speed);

        // angular turn in z direction
        void turn(float duration, float speed);

    private:
        // stop sending messages
        void commonEnd();

        // publish message for duration
        void publishDuration(geometry_msgs::Twist vel, float duration);

        // publish message
        void publishMessage(geometry_msgs::Twist vel);

        ros::Publisher vel_pub_;
};

#endif  // MOTION_UTILITIES.H
