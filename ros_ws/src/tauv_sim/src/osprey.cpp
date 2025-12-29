#include "tauv_sim/osprey.h"

#undef Max  // stonefish opengl Max conflicts with ROS

#include <StonefishCommon.h>
#include <core/FeatherstoneRobot.h>
#include <entities/Entity.h>
#include <entities/SolidEntity.h>
#include <entities/solids/Polyhedron.h>

#include <Eigen/Dense>

#include "tauv_sim/config.h"
#include "tauv_sim/util.h"

using namespace config::osprey;

Osprey::Osprey(const std::string prefix,
               const std::string& assets_path,
               const std::string& hull_material,
               const std::string& hull_look,
               rclcpp::Node::SharedPtr node,
               ConfigLoader& config_loader)
    : prefix_(prefix) {
    sf_robot_ = new sf::FeatherstoneRobot("osprey");

    auto frames = config_loader.get_frames();
    auto body_T_cad = frames.cad_T_body.inverse();
    base_link_ = new sf::Polyhedron("osprey_base_link",
                                    sf::PhysicsSettings(),
                                    assets_path + "/osprey/osprey.obj",
                                    1.0F,
                                    body_T_cad,
                                    assets_path + "/osprey/osprey.obj",
                                    1.0F,
                                    body_T_cad,
                                    hull_material,
                                    hull_look);

    auto inertial_buoyancy_params = config_loader.get_inertial_buoyancy_params();
    auto body_R_cad = sf::Matrix3{frames.cad_T_body.getRotation().inverse()};
    auto [body_T_CG, I_CG] = get_sf_inertia(inertial_buoyancy_params, body_R_cad);

    base_link_->SetArbitraryPhysicalProperties(inertial_buoyancy_params.mass, I_CG, body_T_CG);

    sf_robot_->DefineLinks(base_link_);

    /* Sensors */
    /** Depth **/
    auto depth_params = config_loader.get_depth_params();
    auto depth_sensor = new sf::Pressure("pressure_sensor", depth_params.update_rate);
    depth_sensor->setNoise(depth_params.noise_std);
    depth_sensor->setRange(200'000);

    sf::Transform body_T_depth = sf::Transform{sf::I3(), frames.t_depth_B};
    sf_robot_->AddLinkSensor(depth_sensor, "osprey_base_link", body_T_depth);

    // Add publishers
    auto pressure_pub = node->create_publisher<tauv_msgs::msg::Pressure>(prefix_ + "/pressure", 10);

    // Initialize bridges
    pressure_sensor_bridge_ =
        std::make_unique<PressureSensorBridge>(depth_sensor, pressure_pub, "pressure_link");
}

void Osprey::on_step(const Context& ctx) { pressure_sensor_bridge_->on_step(ctx); }

sf::FeatherstoneRobot* Osprey::get_stonefish_robot() { return sf_robot_; }

std::pair<sf::Transform, sf::Vector3> compute_stonefish_inertia(const InertialBuoyancy& cfg) {
    Eigen::Matrix3d hull_inertia_COM_C;
}
