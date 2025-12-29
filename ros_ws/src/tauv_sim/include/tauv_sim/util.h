#pragma once

#include <Stonefish/StonefishCommon.h>

#include <Eigen/Dense>

#undef Max

#include "tauv_sim/config.h"

Eigen::Matrix3d sf_to_eigen_matrix(const sf::Matrix3& m);

sf::Matrix3 eigen_to_sf_matrix(const Eigen::Matrix3d& m);

std::pair<sf::Transform, sf::Vector3> get_sf_inertia(const config::osprey::InertialBuoyancy& cfg,
                                                     sf::Matrix3 body_R_cad);
