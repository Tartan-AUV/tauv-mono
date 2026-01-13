#pragma once

#include <actuators/Thruster.h>

#include <chrono>
#include <cstdint>
#include <memory>

#include <rclcpp/node.hpp>

#include "tauv_sim/config.h"
#include "tauv_msgs/msg/esc_telemetry.hpp"
#include "tauv_msgs/msg/thruster_setpoint.hpp"
#include "tauv_sim/context.h"

class ThrusterBridge {
   public:
    ThrusterBridge(sf::Thruster* thruster,
                   rclcpp::Publisher<tauv_msgs::msg::EscTelemetry>::SharedPtr pub,
                   double telemetry_rate,
                   uint8_t thruster_esc_id,
                   const config::osprey::actuators::Thrusters& cfg);

    void on_step(const Context& ctx);

    void callback(tauv_msgs::msg::ThrusterSetpoint msg);

   private:
    sf::Thruster* thruster_;
    rclcpp::Publisher<tauv_msgs::msg::EscTelemetry>::SharedPtr pub_;
    const std::chrono::duration<double, std::nano> period_ns_;
    const uint8_t thruster_esc_id_;
    SimTime prev_pub_time_ = std::chrono::seconds{0};
    const config::osprey::actuators::Thrusters& c_;
};
