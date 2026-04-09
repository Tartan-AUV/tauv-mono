#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/node.hpp>
#include <memory>
#include <mutex>
#include <deque>
#include <cmath>
#include <string>
#include <tuple>

class TrajectoryManagerNode : public rclcpp::Node {
public:
    // Change this to adjust how often /desired_state is published
    static constexpr double PUBLISH_RATE_HZ = 20.0;

    // Countdown duration before trajectory execution begins
    static constexpr int COUNTDOWN_SECONDS = 15;

    // Default goal tolerances (used when a setpoint omits them in the YAML)
    static constexpr double DEFAULT_POSITION_TOLERANCE = 0.05;    // meters
    static constexpr double DEFAULT_ANGULAR_TOLERANCE  = M_PI / 12.0; // radians (~15 deg)

    struct Setpoint {
        // Target position (meters, world/odom frame)
        double x, y, z;
        // Target orientation (radians, ZYX convention)
        double roll, pitch, yaw;
        // Desired linear velocity (m/s)
        double vx, vy, vz;
        // Desired angular velocity (rad/s)
        double wx, wy, wz;
        // Goal tolerances
        double position_tolerance = DEFAULT_POSITION_TOLERANCE;
        double angular_tolerance  = DEFAULT_ANGULAR_TOLERANCE;
    };

    TrajectoryManagerNode();
    void addSetpoint(Setpoint setpoint);
    void clearSetpoints();
    virtual ~TrajectoryManagerNode() noexcept;

private:
    std::deque<Setpoint> trajectorySetpoints;
    std::mutex           trajectorySetpointsMutex;

    nav_msgs::msg::Odometry currentOdometry;
    bool hasOdometry    = false;
    bool countdownActive = true;
    int  countdownRemaining = COUNTDOWN_SECONDS;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometrySubscription;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr    desiredStatePublisher;
    rclcpp::TimerBase::SharedPtr                             publishTimer;
    rclcpp::TimerBase::SharedPtr                             countdownTimer;

    void loadSetpointsFromFile(const std::string& path);
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void publishTimerCallback();
    void checkGoalReached();
    void countdownTimerCallback();

    static std::tuple<double, double, double, double> eulerToQuaternion(
        double roll, double pitch, double yaw);
    static std::tuple<double, double, double> quaternionToEuler(
        double x, double y, double z, double w);
    static double wrapAngle(double angle);
};

int main(int argc, char** argv);
