#include "tauv_mission/trajectory_manager.hpp"

#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

TrajectoryManagerNode::TrajectoryManagerNode() : Node("trajectory_manager") {
    // Declare parameter for setpoints file path.
    // Default: <package_share>/config/setpoints.yaml
    // Override via: --ros-args -p setpoints_file:=/path/to/your/setpoints.yaml
    std::string defaultPath =
        ament_index_cpp::get_package_share_directory("tauv_mission") +
        "/config/setpoints.yaml";

    this->declare_parameter<std::string>("setpoints_file", defaultPath);
    std::string setpointsFile = this->get_parameter("setpoints_file").as_string();

    odometrySubscription = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered",
        10,
        std::bind(&TrajectoryManagerNode::odometryCallback, this, std::placeholders::_1)
    );

    desiredStatePublisher = this->create_publisher<nav_msgs::msg::Odometry>(
        "/desired_state",
        10
    );

    publishTimer = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / PUBLISH_RATE_HZ),
        std::bind(&TrajectoryManagerNode::publishTimerCallback, this)
    );

    countdownTimer = this->create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&TrajectoryManagerNode::countdownTimerCallback, this)
    );

    loadSetpointsFromFile(setpointsFile);

    RCLCPP_INFO(this->get_logger(), "--- Trajectory starting in %d seconds ---", COUNTDOWN_SECONDS);
}

void TrajectoryManagerNode::countdownTimerCallback() {
    if (countdownRemaining > 0) {
        RCLCPP_INFO(this->get_logger(), "Trajectory starting in %d...", countdownRemaining);
        countdownRemaining--;
    } else {
        RCLCPP_INFO(this->get_logger(), "--- GO: Trajectory execution started ---");
        countdownActive = false;
        countdownTimer->cancel();
    }
}

void TrajectoryManagerNode::loadSetpointsFromFile(const std::string& path) {
    RCLCPP_INFO(this->get_logger(), "Loading setpoints from: %s", path.c_str());

    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load setpoints file: %s", e.what());
        return;
    }

    if (!root["setpoints"] || !root["setpoints"].IsSequence()) {
        RCLCPP_ERROR(this->get_logger(), "YAML file must contain a 'setpoints' sequence.");
        return;
    }

    std::lock_guard<std::mutex> lock(trajectorySetpointsMutex);
    trajectorySetpoints.clear();

    for (const auto& node : root["setpoints"]) {
        Setpoint sp;

        try {
            sp.x     = node["x"].as<double>();
            sp.y     = node["y"].as<double>();
            sp.z     = node["z"].as<double>();
            sp.roll  = node["roll"].as<double>();
            sp.pitch = node["pitch"].as<double>();
            sp.yaw   = node["yaw"].as<double>();
            sp.vx    = node["vx"].as<double>();
            sp.vy    = node["vy"].as<double>();
            sp.vz    = node["vz"].as<double>();
            sp.wx    = node["wx"].as<double>();
            sp.wy    = node["wy"].as<double>();
            sp.wz    = node["wz"].as<double>();

            // Optional tolerance fields — use defaults if not specified
            sp.position_tolerance = node["position_tolerance"]
                ? node["position_tolerance"].as<double>()
                : DEFAULT_POSITION_TOLERANCE;
            sp.angular_tolerance = node["angular_tolerance"]
                ? node["angular_tolerance"].as<double>()
                : DEFAULT_ANGULAR_TOLERANCE;
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(),
                "Skipping malformed setpoint (missing required field): %s", e.what());
            continue;
        }

        trajectorySetpoints.push_back(sp);
        RCLCPP_INFO(this->get_logger(),
            "  Loaded setpoint: pos=(%.2f, %.2f, %.2f) rpy=(%.2f, %.2f, %.2f) "
            "vel=(%.2f, %.2f, %.2f) ang_vel=(%.2f, %.2f, %.2f)",
            sp.x, sp.y, sp.z,
            sp.roll, sp.pitch, sp.yaw,
            sp.vx, sp.vy, sp.vz,
            sp.wx, sp.wy, sp.wz);
    }

    RCLCPP_INFO(this->get_logger(),
        "Loaded %zu setpoint(s).", trajectorySetpoints.size());
}

void TrajectoryManagerNode::addSetpoint(Setpoint setpoint) {
    std::lock_guard<std::mutex> lock(trajectorySetpointsMutex);
    trajectorySetpoints.push_back(setpoint);
    RCLCPP_INFO(this->get_logger(),
        "Setpoint added: pos=(%.2f, %.2f, %.2f) rpy=(%.2f, %.2f, %.2f) "
        "vel=(%.2f, %.2f, %.2f) ang_vel=(%.2f, %.2f, %.2f)",
        setpoint.x, setpoint.y, setpoint.z,
        setpoint.roll, setpoint.pitch, setpoint.yaw,
        setpoint.vx, setpoint.vy, setpoint.vz,
        setpoint.wx, setpoint.wy, setpoint.wz);
}

void TrajectoryManagerNode::clearSetpoints() {
    std::lock_guard<std::mutex> lock(trajectorySetpointsMutex);
    trajectorySetpoints.clear();
    RCLCPP_INFO(this->get_logger(), "Setpoints cleared.");
}

void TrajectoryManagerNode::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    currentOdometry = *msg;
    hasOdometry = true;
    if (!countdownActive) {
        checkGoalReached();
    }
}

void TrajectoryManagerNode::publishTimerCallback() {
    if (countdownActive) {
        return;
    }

    std::lock_guard<std::mutex> lock(trajectorySetpointsMutex);

    if (trajectorySetpoints.empty()) {
        return;
    }

    const Setpoint& sp = trajectorySetpoints.front();

    nav_msgs::msg::Odometry desired;
    desired.header.stamp    = this->now();
    desired.header.frame_id = "odom";

    // Position: x, y, z
    desired.pose.pose.position.x = sp.x;
    desired.pose.pose.position.y = sp.y;
    desired.pose.pose.position.z = sp.z;

    // Orientation: roll, pitch, yaw -> quaternion
    auto [qx, qy, qz, qw] = eulerToQuaternion(sp.roll, sp.pitch, sp.yaw);
    desired.pose.pose.orientation.x = qx;
    desired.pose.pose.orientation.y = qy;
    desired.pose.pose.orientation.z = qz;
    desired.pose.pose.orientation.w = qw;

    // Linear velocity: vx, vy, vz
    desired.twist.twist.linear.x = sp.vx;
    desired.twist.twist.linear.y = sp.vy;
    desired.twist.twist.linear.z = sp.vz;

    // Angular velocity: wx, wy, wz
    desired.twist.twist.angular.x = sp.wx;
    desired.twist.twist.angular.y = sp.wy;
    desired.twist.twist.angular.z = sp.wz;

    desiredStatePublisher->publish(desired);
}

void TrajectoryManagerNode::checkGoalReached() {
    std::lock_guard<std::mutex> lock(trajectorySetpointsMutex);

    if (trajectorySetpoints.empty() || !hasOdometry) {
        return;
    }

    const Setpoint& sp = trajectorySetpoints.front();

    // Position error (Euclidean distance)
    double dx = currentOdometry.pose.pose.position.x - sp.x;
    double dy = currentOdometry.pose.pose.position.y - sp.y;
    double dz = currentOdometry.pose.pose.position.z - sp.z;
    double posError = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Orientation error (per-axis euler difference, each wrapped to [-pi, pi])
    const auto& q = currentOdometry.pose.pose.orientation;
    auto [curRoll, curPitch, curYaw] = quaternionToEuler(q.x, q.y, q.z, q.w);

    double dRoll  = wrapAngle(curRoll  - sp.roll);
    double dPitch = wrapAngle(curPitch - sp.pitch);
    double dYaw   = wrapAngle(curYaw   - sp.yaw);
    double angError = std::sqrt(dRoll*dRoll + dPitch*dPitch + dYaw*dYaw);

    if (posError <= sp.position_tolerance && angError <= sp.angular_tolerance) {
        RCLCPP_INFO(this->get_logger(),
            "Setpoint reached (pos_err=%.3f m, ang_err=%.3f rad). Moving to next.",
            posError, angError);
        trajectorySetpoints.pop_front();

        if (trajectorySetpoints.empty()) {
            RCLCPP_INFO(this->get_logger(), "Trajectory complete. Waiting for more setpoints.");
        }
    }
}

std::tuple<double, double, double, double> TrajectoryManagerNode::eulerToQuaternion(
    double roll, double pitch, double yaw)
{
    double cr = std::cos(roll  * 0.5);
    double sr = std::sin(roll  * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cy = std::cos(yaw   * 0.5);
    double sy = std::sin(yaw   * 0.5);

    double w = cr * cp * cy + sr * sp * sy;
    double x = sr * cp * cy - cr * sp * sy;
    double y = cr * sp * cy + sr * cp * sy;
    double z = cr * cp * sy - sr * sp * cy;
    return {x, y, z, w};
}

std::tuple<double, double, double> TrajectoryManagerNode::quaternionToEuler(
    double x, double y, double z, double w)
{
    double roll  = std::atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y));
    double pitch = std::asin(std::clamp(2.0 * (w*y - z*x), -1.0, 1.0));
    double yaw   = std::atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z));
    return {roll, pitch, yaw};
}

double TrajectoryManagerNode::wrapAngle(double angle) {
    while (angle >  M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

TrajectoryManagerNode::~TrajectoryManagerNode() noexcept = default;

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajectoryManagerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
