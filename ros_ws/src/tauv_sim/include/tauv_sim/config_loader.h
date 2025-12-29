#pragma once

#include <rclcpp/rclcpp.hpp>
#include <utility>

#include "tauv_sim/config.h"

class ConfigLoader {
   public:
    explicit ConfigLoader(rclcpp::Node::SharedPtr node) : node_(std::move(node)) {}

    void declare_all_parameters();

    config::osprey::Frames get_frames();

    config::osprey::InertialBuoyancy get_inertial_buoyancy_params();

    config::osprey::sensors::Depth get_depth_params();

   private:
    rclcpp::Node::SharedPtr node_;

    void declare_transform(const std::string& ns, const std::string& to, const std::string& from);

    void declare_vector(const std::string& ns, const std::string& name);

    void declare_scalar(const std::string& ns, const std::string& name);

    std::pair<std::string, std::string> get_transform_name(const std::string& ns,
                                                           const std::string& to,
                                                           const std::string& from);

    std::vector<double> get_vector(const std::string& ns, const std::string& name);

    template <size_t N>
    std::array<double, N> get_array(const std::string& ns, const std::string& name);

    sf::Matrix3 get_matrix3(const std::string& ns, const std::string& name);

    sf::Vector3 get_vector3(const std::string& ns, const std::string& name);

    sf::Transform get_transform(const std::string& ns,
                                const std::string& to,
                                const std::string& from);

    double get_scalar(const std::string& ns, const std::string& name);
};
