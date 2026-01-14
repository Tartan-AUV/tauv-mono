#include "tauv_sim/config_loader.h"

#include <Eigen/Dense>
#include <cassert>

#include "tauv_sim/util.h"

using namespace config;

osprey::Frames ConfigLoader::get_frames() {
    const auto ns = std::string{osprey::Frames::NS};

    auto cad_T_body = get_transform(ns, "cad", "body", false);
    auto t_depth_B = get_vector3(ns, "t_depth_B");

    return {cad_T_body, t_depth_B};
}

osprey::InertialBuoyancy ConfigLoader::get_inertial_buoyancy_params() {
    const auto ns = std::string{osprey::InertialBuoyancy::NS};

    auto mass = get_scalar<double>(ns, "mass");
    auto t_hull_com_C = get_vector3(ns, "t_hull_com_C");
    auto hull_inertia_COM_C = get_matrix3(ns, "hull_inertia_COM_C");
    auto t_hull_cob_C = get_vector3(ns, "t_hull_cob_C");

    return {mass, t_hull_com_C, hull_inertia_COM_C, t_hull_cob_C};
}

osprey::sensors::Depth ConfigLoader::get_depth_params() {
    const auto ns = std::string{osprey::sensors::Depth::NS};

    double noise_std = get_scalar<double>(ns, "noise_std");
    double update_rate = get_scalar<double>(ns, "update_rate");

    return {
        noise_std,
        update_rate,
    };
}

osprey::actuators::Thrusters ConfigLoader::get_thrusters() {
    const auto ns = std::string{osprey::actuators::Thrusters::NS};
    const auto n_thrusters = osprey::actuators::Thrusters::N_THRUSTERS;

    osprey::actuators::Thrusters t;

    t.v_bat = get_scalar<double>(ns, "v_bat");
    t.deadband_low = get_scalar<double>(ns, "deadband_low");
    t.deadband_high = get_scalar<double>(ns, "deadband_high");
    t.J_msp = get_scalar<double>(ns, "J_msp");
    t.K_v1 = get_scalar<double>(ns, "K_v1");
    t.K_v2 = get_scalar<double>(ns, "K_v2");
    t.K_t = get_scalar<double>(ns, "K_t");
    t.R_m = get_scalar<double>(ns, "R_m");
    t.K_F_fwd = get_scalar<double>(ns, "K_F_fwd");
    t.K_F_rev = get_scalar<double>(ns, "K_F_rev");
    t.telemetry_rate = get_scalar<double>(ns, "telemetry_rate");

    auto right_handed_int = get_array<long, n_thrusters>(ns, "right_handed");
    std::transform(right_handed_int.begin(),
                   right_handed_int.end(),
                   t.right_handed.begin(),
                   [](long i) { return i != 0; });

    auto esc_thruster_ids_int = get_array<long, n_thrusters>(ns, "esc_thruster_ids");
    std::transform(esc_thruster_ids_int.begin(),
                   esc_thruster_ids_int.end(),
                   t.esc_thruster_ids.begin(),
                   [](long i) { return static_cast<uint8_t>(i); });

    for (int i = 0; i < n_thrusters; ++i) {
        auto thruster_i_frame = "thruster_" + std::to_string(i);
        auto cad_T_thruster_i = get_transform(ns, "cad", thruster_i_frame);
        t.cad_T_thrusters[i] = cad_T_thruster_i;
    }

    return t;
}

std::pair<std::string, std::string> ConfigLoader::get_transform_name(const std::string& ns,
                                                                     const std::string& to,
                                                                     const std::string& from,
                                                                     bool expect_euler) {
    assert(std::isalnum(ns.back()));

    std::string rotation_name = expect_euler ? "rpy_" + to + "__" + from : to + "_R_" + from;
    std::string translation_name = "t_" + to + "__" + from;

    return {rotation_name, translation_name};
}

sf::Transform ConfigLoader::get_transform(const std::string& ns,
                                          const std::string& to,
                                          const std::string& from,
                                          bool expect_euler) {
    auto [rotation_name, translation_name] = get_transform_name(ns, to, from, expect_euler);
    auto t = get_vector3(ns, translation_name);

    if (expect_euler) {
        auto rpy = get_array<double, 3>(ns, rotation_name);

        for (auto& v : rpy) {
            if (!(-180.0 < v && v <= 180.0)) {
                throw std::runtime_error("Invalid rotation angle in " + rotation_name);
            }
        }

        auto Rx = Eigen::AngleAxisd{rpy[0] * M_PI / 180.0, Eigen::Vector3d::UnitX()};
        auto Ry = Eigen::AngleAxisd{rpy[1] * M_PI / 180.0, Eigen::Vector3d::UnitY()};
        auto Rz = Eigen::AngleAxisd{rpy[2] * M_PI / 180.0, Eigen::Vector3d::UnitZ()};

        auto R = Rz * Ry * Rx;
        auto sf_quat = sf::Quaternion(R.x(), R.y(), R.z(), R.w());

        return sf::Transform{sf_quat, t};
    }

    auto raw_rotation = get_matrix3(ns, rotation_name);
    Eigen::Matrix3d R = sf_to_eigen_matrix(raw_rotation);
    std::cout << R << std::endl;
    Eigen::JacobiSVD svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto U = svd.matrixU();
    auto Vt = svd.matrixV().transpose();
    Eigen::Matrix3d R_orthonormal = U * Vt;
    if (R_orthonormal.determinant() < 0.0) {
        U.col(0) = -U.col(0);
        R_orthonormal = U * Vt;
    }

    std::cout << R_orthonormal << std::endl;
    Eigen::Quaterniond q(R_orthonormal);
    std::cout << q << std::endl;
    return sf::Transform{sf::Quaternion{q.x(), q.y(), q.z(), q.w()}, t};
}

std::vector<double> ConfigLoader::get_vector(const std::string& ns, const std::string& name) {
    assert(std::isalnum(ns.back()));

    std::string qualified_name = ns + "." + name;

    std::vector<double> v;
    bool exists = node_->get_parameter(qualified_name, v);

    if (!exists) {
        throw std::runtime_error("Parameter does not exist: " + qualified_name);
    }

    return v;
}

template <typename T, size_t N>
std::array<T, N> ConfigLoader::get_array(const std::string& ns, const std::string& name) {
    assert(std::isalnum(ns.back()));

    std::string qualified_name = ns + "." + name;
    std::vector<T> v;
    v.reserve(N);

    bool exists = node_->get_parameter<std::vector<T>>(qualified_name, v);

    if (!exists) {
        throw std::runtime_error("Parameter does not exist: " + qualified_name);
    }
    if (v.size() != N) {
        throw std::runtime_error("Size mismatch: " + qualified_name);
    }

    auto a = std::array<T, N>{};
    std::copy(v.begin(), v.end(), a.begin());

    return a;
}

sf::Matrix3 ConfigLoader::get_matrix3(const std::string& ns, const std::string& name) {
    auto a = get_array<double, 9>(ns, name);
    auto m = sf::Matrix3{
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7],
        a[8],
    };
    return m;
}

sf::Vector3 ConfigLoader::get_vector3(const std::string& ns, const std::string& name) {
    auto a = get_array<double, 3>(ns, name);
    auto v = sf::Vector3{a[0], a[1], a[2]};
    return v;
}

template <typename T>
T ConfigLoader::get_scalar(const std::string& ns, const std::string& name) {
    assert(std::isalnum(ns.back()));
    std::string qualified_name = ns + "." + name;
    T x;
    bool exists = node_->get_parameter(qualified_name, x);
    if (!exists) {
        throw std::runtime_error("Parameter does not exist: " + qualified_name);
    }
    return x;
}
