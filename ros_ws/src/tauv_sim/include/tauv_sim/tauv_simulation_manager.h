#pragma once

#include <core/SimulationManager.h>
#include <entities/statics/Obstacle.h>

#undef Max

#include <optional>
#include <rclcpp/node.hpp>
#include <string>

#include "tauv_sim/config_loader.h"
#include "tauv_sim/context.h"
#include "tauv_sim/kinematic_osprey.h"
#include "tauv_sim/osprey.h"

class TauvSimulationManager : public sf::SimulationManager {
   public:
    TauvSimulationManager(std::string assets_path,
                          float step_per_second,
                          std::optional<std::string> kinematic_trajectory_path = std::nullopt,
                          bool enable_cameras = true);

    void BuildScenario() override;

    void SimulationStepCompleted(sf::Scalar time_step) override;

   private:
    rclcpp::executors::SingleThreadedExecutor executor_;
    rclcpp::Node::SharedPtr node_;

    std::shared_ptr<ConfigLoader> config_loader_;

    std::unique_ptr<Osprey> robot_ = nullptr;
    std::unique_ptr<KinematicOsprey> kinematic_robot_ = nullptr;
    const std::string assets_path;
    sf::Obstacle* pool_ = nullptr;
    Context ctx_;
    std::optional<std::string> kinematic_trajectory_path_;
    bool enable_cameras_;
};
