#include "tauv_sim/trajectory_test_osprey.h"

#undef Max  // stonefish opengl Max conflicts with ROS

#include <core/SimulationManager.h>
#include <entities/animation/PWLTrajectory.h>

using Pose = geometry_msgs::msg::Pose;
using Point = geometry_msgs::msg::Point;
using GotoVelocity = tauv_msgs::action::GotoVelocity;
using GotoVelocityGoalHandle = rclcpp_action::ServerGoalHandle<tauv_msgs::action::GotoVelocity>;

TrajectoryTestOsprey::TrajectoryTestOsprey(std::string prefix,
                                 const std::string& assets_path,
                                 rclcpp::Node::SharedPtr node,
                                 std::shared_ptr<ConfigLoader> config_loader, 
                                 sf::SimulationManager* sim_manager) : prefix_(std::move(prefix)) {
    auto frames = config_loader->get_frames();
    const sf::Transform body_T_cad = frames.cad_T_body.inverse();

    animated_body_ = std::make_unique<sf::AnimatedEntity>("trajectory_test_osprey",
                                                          new sf::PWLTrajectory(sf::PlaybackMode::ONETIME),
                                                          assets_path + "osprey/hull_visual.stl",
                                                          1.0F,
                                                          body_T_cad,
                                                          assets_path + "osprey/hull_physical.stl",
                                                          1.0F,
                                                          body_T_cad,
                                                          materials::ALUMINUM.name,
                                                          looks::OSPREY_RED_HULL.name,
                                                          false);

    animated_body_->Update(0.0F);

    actionServer = rclcpp_action::create_server<GotoVelocity>(node, "TrajectoryManager", 
      std::bind(&TrajectoryTestOsprey::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&TrajectoryTestOsprey::handleCancel, this, std::placeholders::_1),
      std::bind(&TrajectoryTestOsprey::handleAccepted, this, std::placeholders::_1)
    );

    node_ = node;
    assets_path_ = assets_path;
    body_T_cad_ = body_T_cad;

    sim_manager_ = sim_manager;
}

void TrajectoryTestOsprey::add_to_simulation(sf::SimulationManager* sim_manager) {
    if (animated_body_) {
        sim_manager->AddAnimatedEntity(animated_body_.get());
    }
}

void TrajectoryTestOsprey::on_step(const Context& ctx) { }

sf::AnimatedEntity* TrajectoryTestOsprey::get_entity() { return animated_body_.get(); }

void TrajectoryTestOsprey::goToNewKeypoint(sf::Transform keyTransform){
    const float maxSpeed = 1.5f;

    sf::PWLTrajectory* newTrajectory = new sf::PWLTrajectory(sf::PlaybackMode::ONETIME);

    sf::Transform currentTransform = animated_body_->getCGTransform();
    newTrajectory->AddKeyPoint(0.0f, currentTransform); // make the current transform the first keypoint

    float distance = (currentTransform.getOrigin()).distance(keyTransform.getOrigin());

    newTrajectory->AddKeyPoint(distance/maxSpeed, keyTransform);
    newTrajectory->Play(0.0);
    newTrajectory->Interpolate();


    animated_body_->setTrajectory(newTrajectory);
    animated_body_->Update(0.0f);
}

rclcpp_action::GoalResponse TrajectoryTestOsprey::handleGoal(const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const GotoVelocity::Goal> goal){
    RCLCPP_INFO(node_->get_logger(), "Received goal");
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse TrajectoryTestOsprey::handleCancel(const std::shared_ptr<rclcpp_action::ServerGoalHandle<tauv_msgs::action::GotoVelocity>> goal_handle){
    RCLCPP_INFO(node_->get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
}

bool TrajectoryTestOsprey::withinCurrentTolerance(){
    sf::Transform currentTransform = animated_body_->getCGTransform();
    sf::Vector3 currentPosition = currentTransform.getOrigin();

    bool withinPositionTolerance = (fabs(currentTargetPose.position.x - currentPosition.x()) <= currentPositionTolerance.x &&
                                    fabs(currentTargetPose.position.y - currentPosition.y()) <= currentPositionTolerance.y &&
                                    fabs(currentTargetPose.position.z - currentPosition.z()) <= currentPositionTolerance.z);
    
    sf::Quaternion currentOrientation;
    currentTransform.getBasis().getRotation(currentOrientation);

    float quaternionDotProduct = (currentTargetPose.orientation.x*currentOrientation.x() + currentTargetPose.orientation.y*currentOrientation.y() +
                                  currentTargetPose.orientation.z*currentOrientation.z() + currentTargetPose.orientation.w*currentOrientation.w());
    
    float angleDifference = 2*acos(quaternionDotProduct);
    bool withinAngularTolerance = (angleDifference < currentAngularTolerance);

    return withinPositionTolerance && withinAngularTolerance;
}

void TrajectoryTestOsprey::execute(const std::shared_ptr<GotoVelocityGoalHandle> goal_handle){
    currentTargetPose = goal_handle->get_goal()->target_pose;
    currentPositionTolerance = goal_handle->get_goal()->position_tolerance;
    currentAngularTolerance = goal_handle->get_goal()->angular_tolerance;
    
    Pose requestPose = goal_handle->get_goal()->target_pose;
    geometry_msgs::msg::Quaternion nextOrientation = requestPose.orientation;
    geometry_msgs::msg::Point nextPosition = requestPose.position;

    btQuaternion sfNextOrientation(nextOrientation.x, nextOrientation.y, nextOrientation.z, nextOrientation.w);
    btVector3 sfNextPosition(nextPosition.x, nextPosition.y, nextPosition.z);

    sf::Transform newTransform = sf::Transform(sfNextOrientation, sfNextPosition);
    goToNewKeypoint(newTransform);

    auto feedback = std::make_shared<GotoVelocity::Feedback>();

    // wait until we get there
    while(rclcpp::ok && !withinCurrentTolerance()){
        // Give feedback to mission planning
        sf::Transform currentTransform = animated_body_->getCGTransform();
        sf::Vector3 currentPosition = currentTransform.getOrigin();

        float xDiff = currentPosition.x() - currentTargetPose.position.x;
        float yDiff = currentPosition.y() - currentTargetPose.position.y;
        float zDiff = currentPosition.z() - currentTargetPose.position.z;
        float distanceRemaining = sqrt(xDiff*xDiff + yDiff*yDiff + zDiff*zDiff);
        
        Pose currentPose;
        currentPose.position.set__x(currentPosition.x());
        currentPose.position.set__y(currentPosition.y());
        currentPose.position.set__z(currentPosition.z());

        sf::Quaternion sfOrientation;
        currentTransform.getBasis().getRotation(sfOrientation);

        currentPose.orientation.set__w(sfOrientation.w());
        currentPose.orientation.set__x(sfOrientation.x());
        currentPose.orientation.set__y(sfOrientation.y());
        currentPose.orientation.set__z(sfOrientation.z());

        float currentVelocity = animated_body_->getLinearVelocity().length();

        feedback->current_pose = currentPose;
        feedback->current_velocity = currentVelocity;
        feedback->distance_remaining = distanceRemaining;

        goal_handle->publish_feedback(feedback);
    }

    auto result = std::make_shared<GotoVelocity::Result>();
    goal_handle->succeed(result);
}

void TrajectoryTestOsprey::handleAccepted(const std::shared_ptr<GotoVelocityGoalHandle> goal_handle){
    std::thread{std::bind(&TrajectoryTestOsprey::execute, this, std::placeholders::_1), goal_handle}.detach();   
}