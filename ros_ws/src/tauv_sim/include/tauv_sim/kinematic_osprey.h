#pragma once

#include <entities/animation/PWLTrajectory.h>

#include <memory>
#include <string>

#undef Max

#include "tauv_sim/config_loader.h"
#include "tauv_sim/context.h"
#include "tauv_sim/osprey_sensors.h"
#include "tauv_sim/trajectory_loader.h"

namespace sf {
class SimulationManager;
class AnimatedEntity;
}  // namespace sf

class KinematicOsprey {
   public:
    KinematicOsprey(std::string prefix,
                    const std::string& assets_path,
                    rclcpp::Node::SharedPtr node,
                    std::shared_ptr<ConfigLoader> config_loader,
                    const trajectory::Spec& trajectory_spec,
                    bool enable_cameras = true);

    void add_to_simulation(sf::SimulationManager* sim_manager);
    void on_step(const Context& ctx);

    sf::AnimatedEntity* get_entity();

   private:
    void build_trajectory(const trajectory::Spec& spec);
    std::string prefix_;
    std::unique_ptr<sf::AnimatedEntity> animated_body_;
    std::unique_ptr<sf::PWLTrajectory> trajectory_;
    std::unique_ptr<OspreySensors> sensors_;
};
