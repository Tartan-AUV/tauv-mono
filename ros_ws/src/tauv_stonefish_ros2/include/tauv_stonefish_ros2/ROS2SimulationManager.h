#include <Stonefish/core/SimulationManager.h>

namespace sf {
class ROS2SimulationManager : public SimulationManager {
 public:
  ROS2SimulationManager(double stepsPerSecond);
  void BuildScenario();
};
}  // namespace sf