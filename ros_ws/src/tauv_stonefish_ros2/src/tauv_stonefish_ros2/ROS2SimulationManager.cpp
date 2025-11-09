#include "tauv_stonefish_ros2/ROS2SimulationManager.h"

#include <Stonefish/entities/solids/Sphere.h>
#include <Stonefish/entities/statics/Obstacle.h>
#include <Stonefish/entities/statics/Plane.h>

#include <filesystem>

#include "yaml-cpp/yaml.h"

namespace sf {
ROS2SimulationManager::ROS2SimulationManager(double stepsPerSecond)
    : SimulationManager(stepsPerSecond) {}

void ROS2SimulationManager::BuildScenario() {
  // Physical materials
  CreateMaterial("Aluminium", 2700.0, 0.8);
  CreateMaterial("Steel", 7810.0, 0.9);
  SetMaterialsInteraction("Aluminium", "Aluminium", 0.7, 0.5);
  SetMaterialsInteraction("Steel", "Steel", 0.4, 0.2);
  SetMaterialsInteraction("Aluminium", "Steel", 0.6, 0.4);

  // Graphical materials (looks)
  CreateLook("gray", Color::Gray(0.5f), 0.3f, 0.2f);
  CreateLook("red", Color::RGB(1.f, 0.f, 0.f), 0.1f, 0.f);

  // Create environment
  Plane* plane = new Plane("Ground", 10000.0, "Steel", "gray");
  AddStaticEntity(plane, I4());

  // Create object
  Sphere* sph = new Sphere("Sphere", BodyPhysicsSettings(), 0.1, I4(), "Aluminium", "red");
  AddSolidEntity(sph, Transform(IQ(), Vector3(0.0, 0.0, -1.0)));

  // Create static object
  LoadStaticObjectsFromFile("./src/tauv_stonefish_ros2/data/staticObjects.yaml");
}

void ROS2SimulationManager::LoadStaticObjectsFromFile(std::filesystem::path path) {
  YAML::Node staticObjectsParser = YAML::LoadFile(path);

  for (auto it = staticObjectsParser.begin(); it != staticObjectsParser.end(); it++) {
    YAML::Node object = it->second;

    // object["field"] checks if object contains the key "field" if used in a conditonal
    if (!object["MeshPath"]) {
      printf("Object specified in static object file is missing a mesh!\n");
      continue;
    }

    std::string meshPath = GetNodeParameter<std::string>(object, "MeshPath", "");

    // if the path is local, join it to the directory of the yaml file we're reading
    if (meshPath[0] == '.') {
      std::filesystem::path parent = path.parent_path();
      parent.append(meshPath);
      meshPath = parent;
    }

    Vector3 translation = GetNodeParameter<Vector3>(object, "Translation", Vector3(0, 0, 0));

    Quaternion rotation = IQ();
    Vector3 euler = GetNodeParameter<Vector3>(object, "EulerAngles", Vector3(0, 0, 0));
    rotation.setEulerZYX(euler.z(), euler.y(), euler.x());

    float scale = GetNodeParameter<float>(object, "Scale", 1);

    Transform objectTransform = Transform(rotation, translation);

    // it->first contains the "key"/name of the current Node
    Obstacle* staticObject =
        new Obstacle(it->first.as<std::string>().c_str(), meshPath, scale, objectTransform,
                     meshPath, scale, objectTransform, false, "Steel", "gray");

    AddStaticEntity(staticObject, I4());
  }
}

}  // namespace sf