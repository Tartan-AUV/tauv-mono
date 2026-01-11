#pragma once

#include <core/FeatherstoneRobot.h>
#include <entities/solids/Polyhedron.h>

#include <string>

#undef Max

#include "tauv_sim/config.h"
#include "tauv_sim/config_loader.h"
#include "tauv_sim/pressure_sensor_bridge.h"
#include "tauv_sim/thruster_bridge.h"

class Osprey {
   public:
    Osprey(const std::string prefix,
           const std::string& assets_path,
           rclcpp::Node::SharedPtr node,
           ConfigLoader& config_loader);
    ~Osprey() = default;

    sf::FeatherstoneRobot* get_stonefish_robot();

    void on_step(const Context& ctx);

   private:
    std::string prefix_;
    sf::Polyhedron* base_link_;
    std::shared_ptr<sf::FeatherstoneRobot> construct_robot();

    sf::FeatherstoneRobot* sf_robot_;
    std::unique_ptr<PressureSensorBridge> pressure_sensor_bridge_;

    std::array<std::unique_ptr<ThrusterBridge>, 8> thruster_bridges_;

    sf::Matrix3 compute_principal_inertia_axes(const config::osprey::InertialBuoyancy& cfg);
};