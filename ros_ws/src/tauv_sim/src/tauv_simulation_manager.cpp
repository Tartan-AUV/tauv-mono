#include "tauv_sim/tauv_simulation_manager.h"

#include <entities/statics/Obstacle.h>
#include <graphics/OpenGLDataStructs.h>

#include <cmath>

TauvSimulationManager::TauvSimulationManager(std::string assets_path, float step_per_second)
    : SimulationManager(step_per_second),
      node_(std::make_shared<rclcpp::Node>("tauv_sim")),
      config_loader_(node_),
      assets_path(std::move(assets_path) + "/") {
    executor_.add_node(node_);

    config_loader_.declare_all_parameters();
}

void TauvSimulationManager::BuildScenario() {
    // Ocean configuration
    getMaterialManager()->CreateFluid("OceanWater", 1000.0, 0.002, 1.33);
    EnableOcean(0.0, getMaterialManager()->getFluid("OceanWater"));

    // Solid materials
    CreateMaterial("pool_material", 1000.0, 0.5);
    CreateMaterial("osprey_aluminum", 2700.0, 0.8);

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
                             "pool_material");
    AddStaticEntity(pool_, sf::Transform(sf::Quaternion(0.0, M_PI, 0.0)));

    robot_ = std::make_unique<Osprey>("os",
                                      assets_path,
                                      "osprey_aluminum",
                                      "osprey_red_hull",
                                      node_,
                                      config_loader_);

    AddRobot(robot_->get_stonefish_robot(), sf::I4());

    std::cout << "Done building scenario!\n";
}

void TauvSimulationManager::SimulationStepCompleted(sf::Scalar timeStep) {
    // Update context
    ctx_.sim_time_ = std::chrono::nanoseconds(getSimulationClock() * 1000);

    // Spin the ROS node
    executor_.spin_some();

    // Publish available sensor date and propagate control inputs into teh into the stonefish model
    robot_->on_step(ctx_);
}
