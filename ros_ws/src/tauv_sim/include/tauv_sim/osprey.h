#pragma once

#include <core/FeatherstoneRobot.h>
#include <entities/solids/Polyhedron.h>

#include <array>
#include <memory>
#include <string>

#undef Max

#include <rclcpp/node.hpp>
#include <rclcpp/subscription.hpp>

#include "tauv_msgs/msg/thruster_setpoint.hpp"
#include "tauv_sim/config.h"
#include "tauv_sim/config_loader.h"
#include "tauv_sim/context.h"
#include "tauv_sim/osprey_sensors.h"
#include "tauv_sim/thruster_bridge.h"

class Osprey {
   public:
    Osprey(const std::string prefix,
           const std::string& assets_path,
           rclcpp::Node::SharedPtr node,
           std::shared_ptr<ConfigLoader> config_loader,
           bool enable_cameras = true);
    ~Osprey() = default;

    sf::FeatherstoneRobot* get_stonefish_robot();

    void on_step(const Context& ctx);

   private:
    std::string prefix_;
    sf::Polyhedron* base_link_;
    std::shared_ptr<sf::FeatherstoneRobot> construct_robot();

    sf::FeatherstoneRobot* sf_robot_;
    std::unique_ptr<OspreySensors> sensors_;

    std::array<std::unique_ptr<ThrusterBridge>, 8> thruster_bridges_;
    std::array<std::shared_ptr<rclcpp::Subscription<tauv_msgs::msg::ThrusterSetpoint>>, 8>
        thruster_setpoint_subs_{};

    // Configuration
    config::osprey::actuators::Thrusters thruster_config_;

    sf::Matrix3 compute_principal_inertia_axes(const config::osprey::InertialBuoyancy& cfg);
};
