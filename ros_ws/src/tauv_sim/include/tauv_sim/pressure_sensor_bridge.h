#include <core/FeatherstoneRobot.h>
#include <sensors/scalar/Pressure.h>

#include <optional>

#include "tauv_msgs/msg/pressure.hpp"
#include "tauv_sim/context.h"

class PressureSensorBridge {
   public:
    PressureSensorBridge(sf::Pressure* sensor,
                         rclcpp::Publisher<tauv_msgs::msg::Pressure>::SharedPtr pub,
                         std::string frame_id);

    void on_step(const Context& ctx);

   private:
    sf::Pressure* sensor_;
    const std::string frame_id_;
    rclcpp::Publisher<tauv_msgs::msg::Pressure>::SharedPtr pub_;
};
