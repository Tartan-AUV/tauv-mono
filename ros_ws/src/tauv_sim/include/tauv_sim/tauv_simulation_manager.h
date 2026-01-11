#pragma once

#include <Stonefish/core/SimulationManager.h>
#include <entities/statics/Obstacle.h>

#undef Max

#include <rclcpp/node.hpp>
#include <string>

#include "tauv_sim/config_loader.h"
#include "tauv_sim/context.h"
#include "tauv_sim/osprey.h"

class TauvSimulationManager : public sf::SimulationManager {
   public:
    TauvSimulationManager(std::string assets_path, float step_per_second);

    void BuildScenario() override;

    void SimulationStepCompleted(sf::Scalar time_step) override;

   private:
    rclcpp::executors::SingleThreadedExecutor executor_;
    rclcpp::Node::SharedPtr node_;

    ConfigLoader config_loader_;

    std::unique_ptr<Osprey> robot_ = nullptr;
    const std::string assets_path;
    sf::Obstacle* pool_ = nullptr;
    Context ctx_;
};
