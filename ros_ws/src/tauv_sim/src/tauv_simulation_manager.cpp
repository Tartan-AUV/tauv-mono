#include "tauv_sim/tauv_simulation_manager.h"

#include <entities/statics/Obstacle.h>
#include <graphics/OpenGLDataStructs.h>

#include <cmath>

#include "tauv_sim/registry.h"
#include "tauv_sim/trajectory_loader.h"

TauvSimulationManager::TauvSimulationManager(std::string assets_path,
                                             float step_per_second,
                                             std::optional<std::string> kinematic_trajectory_path)
    : SimulationManager(step_per_second),
      assets_path(std::move(assets_path) + "/"),
      kinematic_trajectory_path_(std::move(kinematic_trajectory_path)) {
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(true);
    options.automatically_declare_parameters_from_overrides(true);

    node_ = std::make_shared<rclcpp::Node>("tauv_sim", options);
    config_loader_ = std::make_shared<ConfigLoader>(node_);

    executor_.add_node(node_);
}

void TauvSimulationManager::BuildScenario() {
    // Ocean configuration
    getMaterialManager()->CreateFluid("OceanWater", 1000.0, 0.002, 1.33);
    EnableOcean(0.0, getMaterialManager()->getFluid("OceanWater"));

    // Materials
    for (const auto& m : materials::all_materials) {
        CreateMaterial(m.name, m.density, m.restitution);
    }

    // Looks
    for (const auto& l : looks::all_looks) {
        CreateLook(l.name, l.color, l.roughness, l.metalness, l.reflectivity);
    }

    // Solid colors
    CreateLook("osprey_red_hull", sf::Color::RGB(1.0f, 0.0f, 0.0f), 0.8f, 0.8f, 0.1f);

    pool_ = new sf::Obstacle("pool",
                             assets_path + "pool/irvine_pool.obj",
                             1.0f,
                             sf::I4(),
                             assets_path + "pool/irvine_pool.obj",
                             1.0f,
                             sf::I4(),
                             false,
                             materials::POOL.name);
    AddStaticEntity(pool_, sf::Transform(sf::Quaternion(0.0, M_PI, 0.0)));

    if (kinematic_trajectory_path_) {
        auto trajectory_spec = trajectory::load_from_yaml(*kinematic_trajectory_path_);
        kinematic_robot_ = std::make_unique<KinematicOsprey>("os",
                                                             assets_path,
                                                             node_,
                                                             config_loader_,
                                                             trajectory_spec);
        kinematic_robot_->add_to_simulation(this);
    } else {
        robot_ = std::make_unique<Osprey>("os", assets_path, node_, config_loader_);
        auto world_T_body_initial = config_loader_->get_initial_pose().world_T_body_initial;
        AddRobot(robot_->get_stonefish_robot(), world_T_body_initial);
    }

    std::cout << "Done building scenario!\n";
}

void TauvSimulationManager::SimulationStepCompleted(sf::Scalar time_step) {
    // Update context
    ctx_.sim_time_ = std::chrono::nanoseconds(getSimulationClock() * 1000);

    // Spin the ROS node
    executor_.spin_some();

    // Publish available sensor date and propagate control inputs into teh into the stonefish model
    if (kinematic_robot_) {
        kinematic_robot_->on_step(ctx_);
    } else if (robot_) {
        robot_->on_step(ctx_);
    }
}
