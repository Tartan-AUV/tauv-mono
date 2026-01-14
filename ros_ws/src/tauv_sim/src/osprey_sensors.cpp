#include "tauv_sim/osprey_sensors.h"

#include <core/FeatherstoneRobot.h>
#include <core/SimulationManager.h>
#include <entities/AnimatedEntity.h>
#include <entities/FeatherstoneEntity.h>

OspreySensors::OspreySensors(std::string prefix,
                             rclcpp::Node::SharedPtr node,
                             std::shared_ptr<ConfigLoader> config_loader,
                             const config::osprey::Frames& frames,
                             const sf::Transform& body_T_cad)
    : prefix_(std::move(prefix)),
      node_(std::move(node)),
      config_loader_(std::move(config_loader)),
      frames_(frames),
      body_T_cad_(body_T_cad) {
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

    auto pressure_pub =
        node_->create_publisher<tauv_msgs::msg::Pressure>(prefix_ + "/sensors/pressure", 10);
    auto imu_pub = node_->create_publisher<sensor_msgs::msg::Imu>(prefix_ + "/sensors/imu", 10);

    pressure_bridge_ = std::make_unique<PressureSensorBridge>(pressure_sensor_.get(),
                                                              pressure_pub,
                                                              "pressure_link");
    imu_bridge_ = std::make_unique<ImuBridge>(imu_sensor_.get(), imu_pub, "imu_link", imu_params);
}

void OspreySensors::attach_to_robot(sf::FeatherstoneRobot* robot) {
    if (!robot) {
        return;
    }

    robot->AddLinkSensor(pressure_sensor_.get(), links::OSPREY_BASE, body_T_depth());
    robot->AddLinkSensor(imu_sensor_.get(), links::OSPREY_BASE, body_T_imu());
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
}

void OspreySensors::on_step(const Context& ctx) {
    if (pressure_bridge_) {
        pressure_bridge_->on_step(ctx);
    }
    if (imu_bridge_) {
        imu_bridge_->on_step(ctx);
    }
}

sf::Transform OspreySensors::body_T_depth() const {
    return sf::Transform{sf::I3(), frames_.t_depth_B};
}

sf::Transform OspreySensors::body_T_imu() const { return body_T_cad_ * frames_.cad_T_imu; }
