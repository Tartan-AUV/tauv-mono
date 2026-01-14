#include "tauv_sim/trajectory_loader.h"

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

sf::PlaybackMode parse_playback_mode(const YAML::Node& root) {
    if (!root["playback_mode"]) {
        return sf::PlaybackMode::ONETIME;
    }

    const auto mode = root["playback_mode"].as<std::string>();
    if (mode == "onetime") {
        return sf::PlaybackMode::ONETIME;
    }
    if (mode == "repeat") {
        return sf::PlaybackMode::REPEAT;
    }
    if (mode == "boomerang") {
        return sf::PlaybackMode::BOOMERANG;
    }

    throw std::runtime_error("Unsupported playback_mode: " + mode +
                             " (expected onetime|repeat|boomerang)");
}

sf::Vector3 to_vector3(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 3) {
        throw std::runtime_error("position must be a 3-element list");
    }
    return sf::Vector3{node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}

Eigen::Matrix3d rpy_to_matrix(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 3) {
        throw std::runtime_error("rpy must be a 3-element list [roll, pitch, yaw]");
    }

    const std::array<double, 3> rpy{node[0].as<double>(),
                                    node[1].as<double>(),
                                    node[2].as<double>()};
    for (auto v : rpy) {
        if (!(-180.0 < v && v <= 180.0)) {
            throw std::runtime_error("Invalid RPY angle (expected degrees in (-180, 180])");
        }
    }

    const auto Rx = Eigen::AngleAxisd{rpy[0] * M_PI / 180.0, Eigen::Vector3d::UnitX()};
    const auto Ry = Eigen::AngleAxisd{rpy[1] * M_PI / 180.0, Eigen::Vector3d::UnitY()};
    const auto Rz = Eigen::AngleAxisd{rpy[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ()};

    return (Rz * Ry * Rx).toRotationMatrix();
}

sf::Quaternion to_quaternion(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond q(R);
    return sf::Quaternion{q.x(), q.y(), q.z(), q.w()};
}

sf::KeyPoint parse_keypoint(const YAML::Node& node) {
    if (!node["t"]) {
        throw std::runtime_error("keyframe is missing required field 't'");
    }
    if (!node["position"]) {
        throw std::runtime_error("keyframe is missing required field 'position'");
    }
    if (!node["rpy"]) {
        throw std::runtime_error("keyframe is missing required field 'rpy'");
    }

    sf::KeyPoint keypoint;
    keypoint.t = node["t"].as<double>();
    const auto position_ned = to_vector3(node["position"]);
    const sf::Vector3 position_enu{position_ned.y(), position_ned.x(), -position_ned.z()};

    const Eigen::Matrix3d ned_R_body = rpy_to_matrix(node["rpy"]);
    const Eigen::Matrix3d enu_R_ned =
        (Eigen::Matrix3d() << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0).finished();
    const Eigen::Matrix3d enu_R_body = enu_R_ned * ned_R_body;

    keypoint.T = sf::Transform{to_quaternion(enu_R_body), position_enu};
    return keypoint;
}

}  // namespace

trajectory::Spec trajectory::load_from_yaml(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);

    if (!root["keyframes"] || !root["keyframes"].IsSequence()) {
        throw std::runtime_error("trajectory YAML must contain a 'keyframes' sequence");
    }

    Spec spec;
    spec.playback_mode = parse_playback_mode(root);
    spec.keypoints.reserve(root["keyframes"].size());

    double previous_time = -1.0;
    for (const auto& keyframe_node : root["keyframes"]) {
        auto keypoint = parse_keypoint(keyframe_node);
        if (keypoint.t < 0.0) {
            throw std::runtime_error("keyframe time must be non-negative");
        }
        if (keypoint.t < previous_time) {
            throw std::runtime_error("keyframe times must be non-decreasing");
        }
        previous_time = keypoint.t;
        spec.keypoints.push_back(keypoint);
    }

    if (spec.keypoints.empty()) {
        throw std::runtime_error("trajectory must contain at least one keyframe");
    }

    return spec;
}
