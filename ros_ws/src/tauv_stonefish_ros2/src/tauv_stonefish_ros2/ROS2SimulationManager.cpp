#include "tauv_stonefish_ros2/ROS2SimulationManager.h"

#include <Stonefish/entities/solids/Sphere.h>
#include <Stonefish/entities/statics/Plane.h>

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
}
}  // namespace sf