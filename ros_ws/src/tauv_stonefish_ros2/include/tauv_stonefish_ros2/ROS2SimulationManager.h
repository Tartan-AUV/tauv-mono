#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/entities/solids/Sphere.h>
#include <Stonefish/entities/statics/Obstacle.h>
#include <Stonefish/entities/statics/Plane.h>

#include <filesystem>

#include "yaml-cpp/yaml.h"

namespace sf {
class ROS2SimulationManager : public SimulationManager {
 public:
  ROS2SimulationManager(double stepsPerSecond);
  void BuildScenario();

 private:
  void LoadStaticObjectsFromFile(std::filesystem::path path);

  // YAML parsing helper
  template <typename T>
  T GetNodeParameter(YAML::Node object, std::string parameterName, T defaultValue) {
    if (object[parameterName]) {
      return object[parameterName].as<T>();
    }
    return defaultValue;
  }
};
}  // namespace sf

// Needed for yaml-cpp's ability to load and serialize stonefish Vector3's
namespace YAML {
template <>
struct convert<btVector3> {
  static Node encode(const btVector3& rhs) {
    Node node;
    node.push_back(rhs.m_floats[0]);
    node.push_back(rhs.m_floats[1]);
    node.push_back(rhs.m_floats[2]);
    return node;
  }

  static bool decode(const Node& node, btVector3& rhs) {
    if (!node.IsSequence() || node.size() != 3) return false;

    rhs.m_floats[0] = node[0].as<float>();
    rhs.m_floats[1] = node[1].as<float>();
    rhs.m_floats[2] = node[2].as<float>();
    return true;
  }
};
}  // namespace YAML