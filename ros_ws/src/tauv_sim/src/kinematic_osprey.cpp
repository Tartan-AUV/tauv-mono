#include "tauv_sim/kinematic_osprey.h"

#undef Max  // stonefish opengl Max conflicts with ROS

#include <core/SimulationManager.h>
#include <entities/animation/PWLTrajectory.h>

KinematicOsprey::KinematicOsprey(std::string prefix,
                                 const std::string& assets_path,
                                 rclcpp::Node::SharedPtr node,
                                 std::shared_ptr<ConfigLoader> config_loader,
                                 const trajectory::Spec& trajectory_spec,
                                 bool enable_cameras)
    : prefix_(std::move(prefix)) {
    auto frames = config_loader->get_frames();
    const sf::Transform body_T_cad = frames.cad_T_body.inverse();

    trajectory_ = std::make_unique<sf::PWLTrajectory>(trajectory_spec.playback_mode);
    build_trajectory(trajectory_spec);

    animated_body_ = std::make_unique<sf::AnimatedEntity>("osprey_kinematic",
                                                          trajectory_.get(),
                                                          assets_path + "osprey/hull_visual.stl",
                                                          1.0F,
                                                          body_T_cad,
                                                          assets_path + "osprey/hull_physical.stl",
                                                          1.0F,
                                                          body_T_cad,
                                                          materials::ALUMINUM.name,
                                                          looks::OSPREY_RED_HULL.name,
                                                          false);

    animated_body_->Update(0.0F);

    sensors_ = std::make_unique<OspreySensors>(prefix_,
                                               node,
                                               config_loader,
                                               frames,
                                               body_T_cad,
                                               enable_cameras);
}

void KinematicOsprey::add_to_simulation(sf::SimulationManager* sim_manager) {
    if (animated_body_) {
        sim_manager->AddAnimatedEntity(animated_body_.get());
    }

    if (sensors_) {
        sensors_->attach_to_animated(animated_body_.get(), sim_manager);
    }
}

void KinematicOsprey::on_step(const Context& ctx) { sensors_->on_step(ctx); }

sf::AnimatedEntity* KinematicOsprey::get_entity() { return animated_body_.get(); }

void KinematicOsprey::build_trajectory(const trajectory::Spec& spec) {
    if (!trajectory_) {
        return;
    }

    for (const auto& keypoint : spec.keypoints) {
        trajectory_->AddKeyPoint(keypoint.t, keypoint.T);
    }

    trajectory_->Play(0.0);
    trajectory_->Interpolate();
}
