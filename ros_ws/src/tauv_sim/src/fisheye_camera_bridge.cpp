#include "tauv_sim/fisheye_camera_bridge.h"

#include <sensor_msgs/image_encodings.hpp>

FisheyeCameraBridge::FisheyeCameraBridge(sf::FisheyeCamera* sensor,
                                         rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub,
                                         std::string frame_id)
    : sensor_(sensor), pub_(std::move(pub)), frame_id_(std::move(frame_id)) {
    if (sensor_) {
        sensor_->getResolution(width_, height_);
    } else {
        width_ = 0;
        height_ = 0;
    }
}

void FisheyeCameraBridge::handle_frame(sf::FisheyeCamera* sensor) {
    if (sensor != sensor_) {
        return;
    }

    auto* data = static_cast<uint8_t*>(sensor->getImageDataPointer());
    if (data == nullptr || width_ == 0 || height_ == 0) {
        return;
    }

    const size_t n_bytes = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3;
    std::lock_guard<std::mutex> lock(mutex_);
    pending_frame_.assign(data, data + n_bytes);
    has_new_frame_ = true;
}

void FisheyeCameraBridge::on_step(const Context& ctx) {
    if (!pub_) {
        return;
    }

    std::vector<uint8_t> frame;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_new_frame_) {
            return;
        }
        frame = std::move(pending_frame_);
        has_new_frame_ = false;
    }

    sensor_msgs::msg::Image msg;
    msg.header.stamp = ctx.get_ros_time();
    msg.header.frame_id = frame_id_;
    msg.height = height_;
    msg.width = width_;
    msg.encoding = sensor_msgs::image_encodings::RGB8;
    msg.is_bigendian = 0;
    msg.step = width_ * 3;
    msg.data = std::move(frame);

    pub_->publish(std::move(msg));
}
