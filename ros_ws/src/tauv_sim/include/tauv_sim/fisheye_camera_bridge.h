#pragma once

#include <sensors/vision/FisheyeCamera.h>
#undef Max  // stonefish defines Max macro that collides with ROS headers

#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <vector>

#include "tauv_sim/context.h"

class FisheyeCameraBridge {
   public:
    FisheyeCameraBridge(sf::FisheyeCamera* sensor,
                        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
                        std::string frame_id);

    // Called from the Stonefish sensor callback.
    void handle_frame(sf::FisheyeCamera* sensor);

    // Publish the most recent frame with the simulation timestamp.
    void on_step(const Context& ctx);

   private:
    sf::FisheyeCamera* sensor_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::string frame_id_;
    unsigned int width_;
    unsigned int height_;

    std::mutex mutex_;
    std::vector<uint8_t> pending_frame_;
    bool has_new_frame_{false};
};
