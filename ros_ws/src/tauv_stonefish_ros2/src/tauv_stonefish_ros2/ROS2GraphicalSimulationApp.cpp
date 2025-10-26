#include "tauv_stonefish_ros2/ROS2GraphicalSimulationApp.h"

#include "tauv_stonefish_ros2/ROS2SimulationManager.h"
#ifdef Max
#undef Max
#endif
#include "rclcpp/rclcpp.hpp"

namespace sf {

ROS2GraphicalSimulationApp::ROS2GraphicalSimulationApp(std::string title, std::string dataPath,
                                                       RenderSettings s, HelperSettings h,
                                                       ROS2SimulationManager* sim)
    : GraphicalSimulationApp(title, dataPath, s, h, sim) {}

void ROS2GraphicalSimulationApp::Startup() {
  Init();
  StartSimulation();
}

void ROS2GraphicalSimulationApp::Tick() {
  LoopInternal();
  if (state_ == SimulationState::FINISHED) {
    CleanUp();
    rclcpp::shutdown();
  }
}

}  // namespace sf