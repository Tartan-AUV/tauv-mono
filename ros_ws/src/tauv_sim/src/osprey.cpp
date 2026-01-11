#include "tauv_sim/osprey.h"

#undef Max  // stonefish opengl Max conflicts with ROS

#include <StonefishCommon.h>
#include <core/FeatherstoneRobot.h>
#include <entities/Entity.h>
#include <entities/SolidEntity.h>
#include <entities/solids/Polyhedron.h>

#include <Eigen/Dense>

#include "tauv_sim/config.h"
#include "tauv_sim/registry.h"
#include "tauv_sim/util.h"

using namespace config::osprey;

Osprey::Osprey(const std::string prefix,
               const std::string& assets_path,
               rclcpp::Node::SharedPtr node,
               ConfigLoader& config_loader)
    : prefix_(prefix) {
    sf_robot_ = new sf::FeatherstoneRobot("osprey");

    auto frames = config_loader.get_frames();
    auto body_T_cad = frames.cad_T_body.inverse();
    base_link_ = new sf::Polyhedron(links::OSPREY_BASE,
                                    sf::PhysicsSettings(),
                                    assets_path + "/osprey/osprey.obj",
                                    1.0F,
                                    body_T_cad,
                                    assets_path + "/osprey/osprey.obj",
                                    1.0F,
                                    body_T_cad,
                                    materials::ALUMINUM.name,
                                    looks::OSPREY_RED_HULL.name);

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
    sf_robot_->AddLinkSensor(depth_sensor, links::OSPREY_BASE, body_T_depth);

    /* Actuators */
    /** Thrusters **/

    auto thruster_config = config_loader.get_thrusters();

    auto prop_physics = sf::PhysicsSettings{};
    prop_physics.buoyancy = false;
    prop_physics.collisions = false;
    prop_physics.mode = sf::PhysicsMode::DISABLED;

    auto rotor_dynamics = std::make_shared<sf::Bessa>(thruster_config.J_msp,
                                                      thruster_config.K_v1,
                                                      thruster_config.K_v2,
                                                      thruster_config.K_t,
                                                      thruster_config.R_m);

    auto thrust_model = std::make_shared<sf::DeadbandThrust>(thruster_config.K_F_rev,
                                                             thruster_config.K_F_fwd,
                                                             0.0F,
                                                             0.0F);

    for (size_t i = 0; i < actuators::Thrusters::N_THRUSTERS; ++i) {
        auto prop = std::make_shared<sf::Polyhedron>("thruster_prop_" + std::to_string(i),
                                                     prop_physics,
                                                     assets_path + "/osprey/prop.obj",
                                                     1.0F,
                                                     sf::I4(),
                                                     materials::PLASTIC.name,
                                                     looks::OSPREY_BLUE_PROP.name);

        auto thruster = new sf::Thruster{"thruster" + std::to_string(i),
                                         prop,
                                         rotor_dynamics,
                                         thrust_model,
                                         0.1F,
                                         thruster_config.right_handed[i],
                                         thruster_config.v_bat,
                                         false,
                                         true};

        auto body_T_thruster = body_T_cad * thruster_config.cad_T_thrusters[i];
        sf_robot_->AddLinkActuator(thruster, links::OSPREY_BASE, body_T_thruster);

        auto setpoint_topic_name =
            prefix_ + "/actuators/thruster_" + std::to_string(i) + "/setpoint";
        auto telemetry_topic_name =
            prefix_ + "/actuators/thruster_" + std::to_string(i) + "/telemetry";

        auto pub = node->create_publisher<tauv_msgs::msg::EscTelemetry>(telemetry_topic_name, 10);
        thruster_bridges_[i] =
            std::make_unique<ThrusterBridge>(thruster,
                                             pub,
                                             thruster_config.telemetry_rate,
                                             thruster_config.esc_thruster_ids[i]);

        node->create_subscription<
            tauv_msgs::msg::ThrusterSetpoint>(setpoint_topic_name,
                                              10,
                                              [&](tauv_msgs::msg::ThrusterSetpoint msg) -> void {
                                                  thruster_bridges_[i]->callback(msg);
                                              });
    };

    // Add publishers
    auto pressure_pub =
        node->create_publisher<tauv_msgs::msg::Pressure>(prefix_ + "/sensors/pressure", 10);

    // Initialize bridges
    pressure_sensor_bridge_ =
        std::make_unique<PressureSensorBridge>(depth_sensor, pressure_pub, "pressure_link");
}

void Osprey::on_step(const Context& ctx) { pressure_sensor_bridge_->on_step(ctx); }

sf::FeatherstoneRobot* Osprey::get_stonefish_robot() { return sf_robot_; }
