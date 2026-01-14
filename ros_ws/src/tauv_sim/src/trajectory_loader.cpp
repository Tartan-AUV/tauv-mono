#include "tauv_sim/trajectory_loader.h"

#include <yaml-cpp/yaml.h>

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

sf::Quaternion to_quaternion(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 4) {
        throw std::runtime_error("quaternion must be a 4-element list [x, y, z, w]");
    }

    const std::array<double, 4> q{node[0].as<double>(),
                                  node[1].as<double>(),
                                  node[2].as<double>(),
                                  node[3].as<double>()};

    const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm < 1e-9) {
        throw std::runtime_error("quaternion has near-zero norm");
    }

    return sf::Quaternion{
        static_cast<sf::Scalar>(q[0] / norm),
        static_cast<sf::Scalar>(q[1] / norm),
        static_cast<sf::Scalar>(q[2] / norm),
        static_cast<sf::Scalar>(q[3] / norm),
    };
}

sf::KeyPoint parse_keypoint(const YAML::Node& node) {
    if (!node["t"]) {
        throw std::runtime_error("keyframe is missing required field 't'");
    }
    if (!node["position"]) {
        throw std::runtime_error("keyframe is missing required field 'position'");
    }
    if (!node["quaternion"]) {
        throw std::runtime_error("keyframe is missing required field 'quaternion'");
    }

    sf::KeyPoint keypoint;
    keypoint.t = node["t"].as<double>();
    keypoint.T = sf::Transform{to_quaternion(node["quaternion"]), to_vector3(node["position"])};
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
