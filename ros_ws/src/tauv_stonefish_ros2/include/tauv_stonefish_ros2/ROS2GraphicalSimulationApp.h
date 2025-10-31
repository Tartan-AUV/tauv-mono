#ifndef ROS2_GRAPHICAL_SIMULATION_APP_H
#define ROS2_GRAPHICAL_SIMULATION_APP_H

#include <Stonefish/core/GraphicalSimulationApp.h>

namespace sf {
class ROS2SimulationManager;

class ROS2GraphicalSimulationApp : public GraphicalSimulationApp {
 public:
  ROS2GraphicalSimulationApp(std::string title, std::string dataPath, RenderSettings s,
                             HelperSettings h, ROS2SimulationManager* sim);
  void Startup();
  void Tick();
};
}  // namespace sf

#endif  // ROS2_GRAPHICAL_SIMULATION_APP_H