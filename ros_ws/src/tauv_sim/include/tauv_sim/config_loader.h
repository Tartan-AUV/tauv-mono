#pragma once

#include <rclcpp/rclcpp.hpp>
#include <utility>

#include "tauv_sim/config.h"

class ConfigLoader {
   public:
    explicit ConfigLoader(rclcpp::Node::SharedPtr node) : node_(std::move(node)) {}

    config::osprey::Frames get_frames();

    config::osprey::InertialBuoyancy get_inertial_buoyancy_params();

    config::osprey::sensors::Depth get_depth_params();

    config::osprey::actuators::Thrusters get_thrusters();

   private:
    rclcpp::Node::SharedPtr node_;

    std::pair<std::string, std::string> get_transform_name(const std::string& ns,
                                                           const std::string& to,
                                                           const std::string& from,
                                                           bool expect_euler);

    std::vector<double> get_vector(const std::string& ns, const std::string& name);

    template <typename T, size_t N>
    std::array<T, N> get_array(const std::string& ns, const std::string& name);

    sf::Matrix3 get_matrix3(const std::string& ns, const std::string& name);

    sf::Vector3 get_vector3(const std::string& ns, const std::string& name);

    sf::Transform get_transform(const std::string& ns,
                                const std::string& to,
                                const std::string& from,
                                bool expect_euler = true);

    template <typename T>
    T get_scalar(const std::string& ns, const std::string& name);
};
