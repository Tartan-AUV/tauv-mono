#include <chrono>
#include <cstdio>

#include "tauv_stonefish_ros2/ROS2GraphicalSimulationApp.h"
#include "tauv_stonefish_ros2/ROS2SimulationManager.h"
#ifdef Max
#undef Max
#endif
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

// Use std::chrono_literals to easily specify time durations (e.g., 1s, 500ms)
using namespace std::chrono_literals;

class SimNode : public rclcpp::Node {
 public:
  SimNode() : Node("sim_node") {
    sf::RenderSettings s;
    s.windowW = 800;
    s.windowH = 600;

    s.shadows = sf::RenderQuality::LOW;
    s.ao = sf::RenderQuality::DISABLED;
    s.atmosphere = sf::RenderQuality::LOW;
    s.ocean = sf::RenderQuality::LOW;
    s.aa = sf::RenderQuality::LOW;
    s.ssr = sf::RenderQuality::DISABLED;

    sf::HelperSettings h;
    h.showFluidDynamics = false;
    h.showCoordSys = false;
    h.showBulletDebugInfo = false;
    h.showSensors = false;
    h.showActuators = false;
    h.showForces = false;

    // other initializations
    sf::ROS2SimulationManager* manager = new sf::ROS2SimulationManager(100);
    app_ = std::shared_ptr<sf::ROS2GraphicalSimulationApp>(
        new sf::ROS2GraphicalSimulationApp("Stonefish Simulator", "", s, h, manager));
    app_->Startup();
    timer_ = this->create_wall_timer(10ms, std::bind(&sf::ROS2GraphicalSimulationApp::Tick, app_));
  }

 private:
  std::shared_ptr<sf::ROS2GraphicalSimulationApp> app_;
  // timer and callback declarations
  //  Declaration of the timer. It's a shared pointer to a TimerBase object.
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimNode>());
  rclcpp::shutdown();
  return 0;
}