#include <core/GraphicalSimulationApp.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#undef Max

#include "tauv_sim/tauv_simulation_manager.h"

namespace {

struct ParsedArgs {
    std::vector<std::string> ros_args;
    std::optional<std::string> kinematic_trajectory_path;
    bool enable_cameras{true};
};

ParsedArgs parse_args(int argc, char** argv) {
    ParsedArgs result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--kinematic" || arg == "-k") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--kinematic requires a file path");
            }
            result.kinematic_trajectory_path = std::string(argv[i + 1]);
            ++i;
            continue;
        }

        if (arg == "--no-cameras" || arg == "--headless") {
            result.enable_cameras = false;
            continue;
        }

        result.ros_args.emplace_back(arg);
    }

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    ParsedArgs parsed;
    try {
        parsed = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Argument error: %s\n", e.what());
        return 1;
    }

    std::vector<char*> ros_argv;
    ros_argv.reserve(parsed.ros_args.size() + 1);
    ros_argv.push_back(argv[0]);
    for (auto& arg : parsed.ros_args) {
        ros_argv.push_back(const_cast<char*>(arg.c_str()));
    }
    int ros_argc = static_cast<int>(ros_argv.size());
    rclcpp::init(ros_argc, ros_argv.data());

    const auto share_path = ament_index_cpp::get_package_share_directory("tauv_sim");
    const std::string assets_path = (std::filesystem::path(share_path) / "assets").string();

    sf::RenderSettings s;
    sf::HelperSettings h;
    h.showActuators = true;
    h.showSensors = true;
    TauvSimulationManager manager(assets_path,
                                  100.0f,
                                  parsed.kinematic_trajectory_path,
                                  parsed.enable_cameras);

    sf::GraphicalSimulationApp app("TAUV Simulator", assets_path, s, h, &manager);
    app.Run();

    return 0;
}
