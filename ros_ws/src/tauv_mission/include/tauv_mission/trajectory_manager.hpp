#pragma once

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <memory>
#include <functional>
#include <sstream>
#include <mutex>

#include "tauv_msgs/action/goto_velocity.hpp"

class TrajectoryManagerNode : public rclcpp::Node{
    public:
        struct Setpoint {
            geometry_msgs::msg::Pose targetPose;
            float velocity;
            float maxTime; // the time it takes before we consider the motion failed and make a new plan
            geometry_msgs::msg::Point positionTolerance;
            float angularTolerance;
        };

        TrajectoryManagerNode();
        void addSetpoint(geometry_msgs::msg::Pose targetPose, float velocity);
        void addSetpoint(geometry_msgs::msg::Pose targetPose, float velocity, float maxTime);
        void addSetpoint(geometry_msgs::msg::Pose targetPose, float velocity, float maxTime, geometry_msgs::msg::Point positionTolerance, float angularTolerance);
        void clearSetpoints();
        virtual ~TrajectoryManagerNode() noexcept;

    private:
        std::deque<Setpoint> trajectorySetpoints; //always tries to reach the front item
        std::mutex trajectorySetpointsMutex; // prevents corruption of trajectorySetpoints, locks in functions which modify trajectorySetpoint's contents
        rclcpp_action::Client<tauv_msgs::action::GotoVelocity>::SharedPtr actionClient;
        rclcpp::TimerBase::SharedPtr retryTimer;

        void sendSetpoint();
        void goalResponseCallback(const rclcpp_action::ClientGoalHandle<tauv_msgs::action::GotoVelocity>::SharedPtr& goalHandle);
        void feedbackCallback(rclcpp_action::ClientGoalHandle<tauv_msgs::action::GotoVelocity>::SharedPtr goalHandle, std::shared_ptr<const tauv_msgs::action::GotoVelocity::Feedback> feedback);
        void resultCallback(const rclcpp_action::ClientGoalHandle<tauv_msgs::action::GotoVelocity>::WrappedResult& result);
};

int main(int argc, char** argv);
geometry_msgs::msg::Quaternion angleAxis(float ax, float ay, float az, float angle);
geometry_msgs::msg::Quaternion angleAxis(geometry_msgs::msg::Point axis, float angle);