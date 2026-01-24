#pragma once

#include <StonefishCommon.h>
#include <entities/animation/PWLTrajectory.h>

#include <memory>
#include <string>

#include <SDL2/SDL_video.h>

#include "tauv_msgs/action/goto_velocity.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#undef Max

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "tauv_sim/config_loader.h"
#include "tauv_sim/context.h"
#include "tauv_sim/osprey_sensors.h"
#include "tauv_sim/trajectory_loader.h"

namespace sf {
class SimulationManager;
class AnimatedEntity;
}  // namespace sf

class TrajectoryTestOsprey {
   public:
    TrajectoryTestOsprey(std::string prefix,
                    const std::string& assets_path,
                    rclcpp::Node::SharedPtr node,
                    std::shared_ptr<ConfigLoader> config_loader,
                    sf::SimulationManager* sim_manager);

    void add_to_simulation(sf::SimulationManager* sim_manager);
    void on_step(const Context& ctx);

    sf::AnimatedEntity* get_entity();

   private:
    void goToNewKeypoint(sf::Transform keyTransform);
    rclcpp_action::GoalResponse handleGoal(const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const tauv_msgs::action::GotoVelocity::Goal> goal);
    rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<rclcpp_action::ServerGoalHandle<tauv_msgs::action::GotoVelocity>> goal_handle);
    void handleAccepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<tauv_msgs::action::GotoVelocity>> goal_handle);
    void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<tauv_msgs::action::GotoVelocity>> goal_handle);
    bool withinCurrentTolerance(void);

    std::string prefix_;
    std::unique_ptr<sf::AnimatedEntity> animated_body_;
    rclcpp_action::Server<tauv_msgs::action::GotoVelocity>::SharedPtr actionServer;
    rclcpp::Node::SharedPtr node_;
    std::string assets_path_;
    sf::Transform body_T_cad_;
    sf::SimulationManager* sim_manager_;

    geometry_msgs::msg::Pose currentTargetPose;
    geometry_msgs::msg::Point currentPositionTolerance;
    float currentAngularTolerance;
};
