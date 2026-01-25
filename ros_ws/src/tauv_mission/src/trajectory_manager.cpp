#include "tauv_mission/trajectory_manager.hpp"

using Pose = geometry_msgs::msg::Pose;
using Point = geometry_msgs::msg::Point;
using Quaternion = geometry_msgs::msg::Quaternion;
using GotoVelocity = tauv_msgs::action::GotoVelocity;
using GotoVelocityGoalHandle = rclcpp_action::ClientGoalHandle<tauv_msgs::action::GotoVelocity>;

TrajectoryManagerNode::TrajectoryManagerNode() : Node("trajectory_manager"), trajectorySetpoints() {
    actionClient = rclcpp_action::create_client<GotoVelocity>(
        this,
        "TrajectoryManager"
    );
}

void TrajectoryManagerNode::addSetpoint(Pose targetPose, float velocity){
    //TODO: Change to something reasonable based on max velocity and distance later on
    addSetpoint(targetPose, velocity, 10000.0f);
}

void TrajectoryManagerNode::addSetpoint(Pose targetPose, float velocity, float maxTime){
    geometry_msgs::msg::Point defaultPositionTolerance;
    defaultPositionTolerance.set__x(0.05).set__y(0.05).set__z(0.05); // default to within 5 cm of target point
    const float defaultAngularTolerance = M_PI/12.0f; // get within PI/12 radians of the target orientation
    
    addSetpoint(
        targetPose,
        velocity,
        maxTime, 
        defaultPositionTolerance, 
        defaultAngularTolerance
    );
}

void TrajectoryManagerNode::addSetpoint(Pose targetPose, float velocity, float maxTime, Point positionTolerance, float angularTolerance){
    std::lock_guard<std::mutex> lock(trajectorySetpointsMutex); // Function modifies trajectorySetpoints, so we lock
    bool startNewRequestChain = trajectorySetpoints.empty();

    trajectorySetpoints.push_back(
        (Setpoint){
            .targetPose = targetPose,
            .velocity = velocity,
            .maxTime = maxTime,
            .positionTolerance = positionTolerance,
            .angularTolerance = angularTolerance,
        }
    );

    if(startNewRequestChain){
        sendSetpoint();
    }
}

void TrajectoryManagerNode::clearSetpoints(){
    trajectorySetpoints.clear();
}

void TrajectoryManagerNode::sendSetpoint(){
    if(trajectorySetpoints.empty()){
        RCLCPP_ERROR(this->get_logger(), "Setpoint Queue Empty!");
        return;
    }

    if(!actionClient->wait_for_action_server(std::chrono::milliseconds{100L})) {
        RCLCPP_ERROR(this->get_logger(), "Action server not available. Trying again in 750ms");
        retryTimer = this->create_wall_timer(
        std::chrono::seconds(2),
        [this]() {
            retryTimer->cancel();
            this->sendSetpoint(); 
        });
        return;
    }

    // Read next setpoint
    Setpoint nextSetpoint = trajectorySetpoints.at(0);

    GotoVelocity::Goal nextGotoVelocity{};
    nextGotoVelocity.target_pose = nextSetpoint.targetPose;
    nextGotoVelocity.velocity = nextSetpoint.velocity;
    nextGotoVelocity.position_tolerance = nextSetpoint.positionTolerance;
    nextGotoVelocity.angular_tolerance = nextSetpoint.angularTolerance;

    // setup callbacks
    rclcpp_action::Client<GotoVelocity>::SendGoalOptions sendOptions{};
    sendOptions.goal_response_callback = std::bind(&TrajectoryManagerNode::goalResponseCallback, this, std::placeholders::_1);
    sendOptions.feedback_callback = std::bind(&TrajectoryManagerNode::feedbackCallback, this, std::placeholders::_1, std::placeholders::_2);
    sendOptions.result_callback = std::bind(&TrajectoryManagerNode::resultCallback, this, std::placeholders::_1);
    
    // and send the next goal
    actionClient->async_send_goal(nextGotoVelocity, sendOptions);
}

void TrajectoryManagerNode::goalResponseCallback(const GotoVelocityGoalHandle::SharedPtr& goalHandle){
    if(goalHandle == nullptr){
        RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
    }
    else{
        RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
    }
}

void TrajectoryManagerNode::feedbackCallback(GotoVelocityGoalHandle::SharedPtr goalHandle, const std::shared_ptr<const GotoVelocityGoalHandle::Feedback> feedback){
    std::stringstream ss;
    ss << "At velocity: " << feedback->current_velocity << " | Distance Remaining: " << feedback->distance_remaining;
    RCLCPP_INFO(this->get_logger(), ss.str().c_str());
}

void TrajectoryManagerNode::resultCallback(const GotoVelocityGoalHandle::WrappedResult & result){
    if (!rclcpp::ok()) {
        return; // Don't try to log or send goals if ROS is shutting down
    }

    std::lock_guard<std::mutex> lock(trajectorySetpointsMutex); // Function modifies trajectorySetpoints, so we lock

    switch (result.code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        // remove the current setpoint
        trajectorySetpoints.pop_front();
        
        // Move to the next setpoint if it exists
        if(!trajectorySetpoints.empty()){
            sendSetpoint();
        }
        else{
            RCLCPP_INFO(this->get_logger(), "Current trajectory complete! Waiting on more setpoints.");
        }
        break;
      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
        return;
      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
        return;
      default:
        RCLCPP_ERROR(this->get_logger(), "Unknown result code");
        return;
    }
}

TrajectoryManagerNode::~TrajectoryManagerNode() noexcept = default;

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    std::shared_ptr<TrajectoryManagerNode> node = std::make_shared<TrajectoryManagerNode>();
    
    std::thread testThread([&node]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        Pose test1{};
        test1.position.set__x(3);
        Pose test2{};
        test2.position.set__y(3);
        test2.set__orientation(angleAxis(0, 0, 1, M_PI/2));
        Pose test3{};
        test3.position.set__z(3);
        node.get()->addSetpoint(test1, 10.0f);
        node.get()->addSetpoint(test2, 10.0f);
        node.get()->addSetpoint(test3, 10.0f);
    });

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

Quaternion angleAxis(float ax, float ay, float az, float angle){
    float length = sqrt(ax*ax + ay*ay + az*az);
    float ux = (ax / length);
    float uy = (ay / length);
    float uz = (az / length);

    Quaternion result{};
    result.set__x(ux * sin(angle/2));
    result.set__y(uy * sin(angle/2));
    result.set__z(uz * sin(angle/2));
    result.set__w(cos(angle/2));

    return result;
}

Quaternion angleAxis(Point axis, float angle){
    return angleAxis(axis.x, axis.y, axis.z, angle);
}