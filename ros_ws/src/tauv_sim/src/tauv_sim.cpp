#include <core/GraphicalSimulationApp.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cstdio>
#include <filesystem>

#undef Max

#include "tauv_sim/tauv_simulation_manager.h"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    rclcpp::init(argc, argv);

    const auto share_path = ament_index_cpp::get_package_share_directory("tauv_sim");
    const std::string assets_path = (std::filesystem::path(share_path) / "assets").string();

    sf::RenderSettings s;
    sf::HelperSettings h;
    TauvSimulationManager manager(assets_path, 100.0f);

    sf::GraphicalSimulationApp app("TAUV Simulator", assets_path, s, h, &manager);
    app.Run();

    return 0;
}
