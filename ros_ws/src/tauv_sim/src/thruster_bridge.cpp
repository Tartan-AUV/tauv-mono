#include "tauv_sim/thruster_bridge.h"

constexpr double RADPS_TO_RPM = 60.0 / (2.0 * M_PI);

ThrusterBridge::ThrusterBridge(sf::Thruster* thruster,
                               rclcpp::Publisher<tauv_msgs::msg::EscTelemetry>::SharedPtr pub,
                               double telemetry_rate,
                               uint8_t thruster_esc_id)
    : thruster_(thruster),
      pub_(pub),
      period_ns_(std::chrono::nanoseconds(static_cast<long>(std::round(1e9 / telemetry_rate)))),
      thruster_esc_id_(thruster_esc_id) {}

void ThrusterBridge::callback(tauv_msgs::msg::ThrusterSetpoint msg) {
    double throttle = msg.enable != 0 ? msg.throttle : 0.0;
    thruster_->setSetpoint(throttle);
}

void ThrusterBridge::on_step(const Context& ctx) {
    using namespace std::chrono_literals;
    if (ctx.sim_time_ - prev_pub_time_ >= period_ns_) {
        prev_pub_time_ = ctx.sim_time_;
        tauv_msgs::msg::EscTelemetry msg;
        msg.header.frame_id = "thruster_" + std::to_string(thruster_esc_id_);
        msg.header.stamp = ctx.get_ros_time();
        msg.id = thruster_esc_id_;
        msg.rpm = thruster_->getOmega() * RADPS_TO_RPM;
        msg.voltage = 0.0;
        msg.current = 0.0;
        msg.temperature = 0.0;
        msg.fault_code = 0U;

        pub_->publish(msg);
    }
}
