#include "tauv_sim/osprey_sensors.h"

#include <core/FeatherstoneRobot.h>
#include <core/SimulationManager.h>
#include <entities/AnimatedEntity.h>
#include <entities/FeatherstoneEntity.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>

OspreySensors::OspreySensors(std::string prefix,
                             rclcpp::Node::SharedPtr node,
                             std::shared_ptr<ConfigLoader> config_loader,
                             const config::osprey::Frames& frames,
                             const sf::Transform& body_T_cad,
                             bool enable_cameras)
    : prefix_(std::move(prefix)),
      node_(std::move(node)),
      config_loader_(std::move(config_loader)),
      frames_(frames),
      body_T_cad_(body_T_cad),
      cameras_enabled_(enable_cameras) {
    const auto depth_params = config_loader_->get_depth_params();
    pressure_sensor_ = std::make_unique<sf::Pressure>("pressure_sensor", depth_params.update_rate);
    pressure_sensor_->setNoise(depth_params.noise_std);
    pressure_sensor_->setRange(200'000);

    const auto imu_params = config_loader_->get_imu_params();
    imu_sensor_ = std::make_unique<sf::IMU>("imu", imu_params.update_rate);
    imu_sensor_->setRange(imu_params.angular_velocity_range, imu_params.linear_acceleration_range);
    imu_sensor_->setNoise(imu_params.angle_std,
                          imu_params.angular_velocity_std,
                          imu_params.yaw_angle_drift,
                          imu_params.linear_acceleration_std);

    const auto dvl_params = config_loader_->get_dvl_params();
    dvl_sensor_ = std::make_unique<sf::DVL>("dvl",
                                            7,
                                            true,
                                            dvl_params.update_rate);  // Stonefish uses NED so I
                                                                      // think true is correct?
    dvl_sensor_->setRange(dvl_params.linear_velocity_range, 0, 10);
    dvl_sensor_->setNoise(dvl_params.linear_velocity_percent_noise,
                          dvl_params.linear_velocity_stddev_noise,
                          0,
                          0,
                          0);

    auto pressure_pub =
        node_->create_publisher<sensor_msgs::msg::FluidPressure>(prefix_ + "/sensors/pressure", 10);
    auto imu_pub = node_->create_publisher<sensor_msgs::msg::Imu>(prefix_ + "/sensors/imu", 10);
    auto dvl_pub = node_->create_publisher<tauv_msgs::msg::Dvl>(prefix_ + "/sensors/dvl", 10);

    pressure_bridge_ = std::make_unique<PressureSensorBridge>(pressure_sensor_.get(),
                                                              pressure_pub,
                                                              prefix_ + "/pressure_link");
    imu_bridge_ =
        std::make_unique<ImuBridge>(imu_sensor_.get(), imu_pub, prefix_ + "/imu_link", imu_params);
    dvl_bridge_ =
        std::make_unique<DvlBridge>(dvl_sensor_.get(), dvl_pub, prefix_ + "/dvl_link", dvl_params);

    if (cameras_enabled_) {
        auto camera_params = config_loader_->get_fisheye_cameras();
        for (size_t i = 0; i < camera_params.size(); ++i) {
            const auto& cam_cfg = camera_params[i];
            cameras_[i] = std::make_unique<sf::FisheyeCamera>("fisheye_" + std::to_string(i),
                                                              cam_cfg.resolution[0],
                                                              cam_cfg.resolution[1],
                                                              cam_cfg.horizontal_fov_deg,
                                                              cam_cfg.update_rate);
            cameras_[i]->setExposure(cam_cfg.exposure);
            cameras_[i]->setDisplayOnScreen(cam_cfg.display_on_screen,
                                            cam_cfg.screen_offset[0],
                                            cam_cfg.screen_offset[1],
                                            static_cast<float>(cam_cfg.screen_scale));

            auto image_pub =
                node_->create_publisher<sensor_msgs::msg::Image>(prefix_ + "/sensors/cam" +
                                                                     std::to_string(i) +
                                                                     "/image_raw",
                                                                 10);
            std::string frame_id = prefix_ + "/cam" + std::to_string(i) + "_optical";
            camera_bridges_[i] =
                std::make_unique<FisheyeCameraBridge>(cameras_[i].get(), image_pub, frame_id);
            cameras_[i]->InstallNewDataHandler([this, i](sf::FisheyeCamera* cam) {
                if (camera_bridges_[i]) {
                    camera_bridges_[i]->handle_frame(cam);
                }
            });
        }
    }

    // Make sure the tf broadcaster is only made once
    static std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster;
    if (!tf_broadcaster)
        tf_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    rclcpp::Time now = node_->get_clock()->now();

    // Turn all the stonefish transforms into tf transforms
    auto add_tf = [&](const sf::Transform& T, const std::string& child_suffix) {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = now;
        t.header.frame_id = prefix_ + "/base_link";
        t.child_frame_id = prefix_ + child_suffix;
        t.transform.translation.x = T.getOrigin().y();
        t.transform.translation.y = T.getOrigin().x();
        t.transform.translation.z = -T.getOrigin().z();
        t.transform.rotation.x = T.getRotation().x();
        t.transform.rotation.y = T.getRotation().y();
        t.transform.rotation.z = T.getRotation().z();
        t.transform.rotation.w = T.getRotation().w();
        tfs.push_back(t);
    };

    add_tf(body_T_depth(), "/pressure_link");
    add_tf(body_T_imu(), "/imu_link");
    add_tf(body_T_dvl(), "/dvl_link");

    if (cameras_enabled_) {
        for (size_t i = 0; i < cameras_.size(); ++i) {
            add_tf(body_T_cam(i), "/cam" + std::to_string(i) + "_optical");
        }
    }

    tf_broadcaster->sendTransform(tfs);
}

void OspreySensors::attach_to_robot(sf::FeatherstoneRobot* robot) {
    if (!robot) {
        return;
    }

    robot->AddLinkSensor(pressure_sensor_.get(), links::OSPREY_BASE, body_T_depth());
    robot->AddLinkSensor(imu_sensor_.get(), links::OSPREY_BASE, body_T_imu());
    if (cameras_enabled_) {
        for (size_t i = 0; i < cameras_.size(); ++i) {
            if (cameras_[i]) {
                robot->AddVisionSensor(cameras_[i].get(), links::OSPREY_BASE, body_T_cam(i));
            }
        }
    }
}

void OspreySensors::attach_to_animated(sf::AnimatedEntity* entity,
                                       sf::SimulationManager* sim_manager) {
    if (!entity || !sim_manager) {
        return;
    }

    pressure_sensor_->AttachToSolid(entity, body_T_depth());
    imu_sensor_->AttachToSolid(entity, body_T_imu());

    sim_manager->AddSensor(pressure_sensor_.get());
    sim_manager->AddSensor(imu_sensor_.get());

    if (cameras_enabled_) {
        for (size_t i = 0; i < cameras_.size(); ++i) {
            if (!cameras_[i]) {
                continue;
            }
            cameras_[i]->AttachToSolid(entity, body_T_cam(i));
            sim_manager->AddSensor(cameras_[i].get());
        }
    }
}

void OspreySensors::on_step(const Context& ctx) {
    if (pressure_bridge_) {
        pressure_bridge_->on_step(ctx);
    }
    if (imu_bridge_) {
        imu_bridge_->on_step(ctx);
    }
    if (cameras_enabled_) {
        for (auto& bridge : camera_bridges_) {
            if (bridge) {
                bridge->on_step(ctx);
            }
        }
    }
}

sf::Transform OspreySensors::body_T_depth() const {
    return sf::Transform{sf::I3(), frames_.t_depth_B};
}

sf::Transform OspreySensors::body_T_dvl() const { return body_T_cad_ * frames_.cad_T_dvl; }

sf::Transform OspreySensors::body_T_imu() const { return body_T_cad_ * frames_.cad_T_imu; }

sf::Transform OspreySensors::body_T_cam(size_t idx) const {
    if (idx == 0) {
        return body_T_cad_ * frames_.cad_T_cam0;
    }
    return body_T_cad_ * frames_.cad_T_cam1;
}
