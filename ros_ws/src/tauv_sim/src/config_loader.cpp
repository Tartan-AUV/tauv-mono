#include "tauv_sim/config_loader.h"

#include <Eigen/Dense>

using namespace config;

void ConfigLoader::declare_all_parameters() {
    std::string ns;

    /* World */

    /* Osprey */
    ns = osprey::Frames::NS;
    declare_transform(ns, "cad", "body");
    declare_vector(ns, "t_depth_B");

    /** Inertial and Buoyancy Parameters **/
    ns = osprey::InertialBuoyancy::NS;
    declare_vector(ns, "hull_inertia_COM_C");
    declare_scalar(ns, "mass");
    declare_vector(ns, "t_hull_com_C");
    declare_vector(ns, "t_hull_cob_C");

    /** Sensors **/
    /*** Depth ***/
    ns = osprey::sensors::Depth::NS;
    declare_scalar(ns, "noise_std");
    declare_scalar(ns, "update_rate");
}

osprey::Frames ConfigLoader::get_frames() {
    const auto ns = std::string{osprey::Frames::NS};

    auto cad_T_body = get_transform(ns, "cad", "body");
    auto t_depth_B = get_vector3(ns, "t_depth_B");

    return {cad_T_body, t_depth_B};
}

osprey::InertialBuoyancy ConfigLoader::get_inertial_buoyancy_params() {
    const auto ns = std::string{osprey::InertialBuoyancy::NS};

    auto mass = get_scalar(ns, "mass");
    auto t_hull_com_C = get_vector3(ns, "t_hull_com_C");
    auto hull_inertia_COM_C = get_matrix3(ns, "hull_inertia_COM_C");
    auto t_hull_cob_C = get_vector3(ns, "t_hull_cob_C");

    return {mass, t_hull_com_C, hull_inertia_COM_C, t_hull_cob_C};
}

osprey::sensors::Depth ConfigLoader::get_depth_params() {
    const auto ns = std::string{osprey::sensors::Depth::NS};

    double noise_std = get_scalar(ns, "noise_std");
    double update_rate = get_scalar(ns, "update_rate");

    return {
        noise_std,
        update_rate,
    };
}

std::pair<std::string, std::string> ConfigLoader::get_transform_name(const std::string& ns,
                                                                     const std::string& to,
                                                                     const std::string& from) {
    assert(std::isalnum(ns.back()));

    std::string rotation_name = ns + ".rpy_" + to + "__" + from;
    std::string translation_name = ns + ".t_" + to + "__" + from;

    return {rotation_name, translation_name};
}

void ConfigLoader::declare_transform(const std::string& ns,
                                     const std::string& to,
                                     const std::string& from) {
    auto [translation_name, rotation_name] = get_transform_name(ns, to, from);
    node_->declare_parameter(rotation_name, rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY);
    node_->declare_parameter(translation_name, rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY);
}

void ConfigLoader::declare_vector(const std::string& ns, const std::string& name) {
    assert(std::isalnum(ns.back()));
    std::string qualified_name = ns + "." + name;
    node_->declare_parameter(qualified_name, rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY);
}

void ConfigLoader::declare_scalar(const std::string& ns, const std::string& name) {
    assert(std::isalnum(ns.back()));
    std::string qualified_name = ns + "." + name;
    node_->declare_parameter(qualified_name, rclcpp::ParameterType::PARAMETER_DOUBLE);
}

sf::Transform ConfigLoader::get_transform(const std::string& ns,
                                          const std::string& to,
                                          const std::string& from) {
    auto [rotation_name, translation_name] = get_transform_name(ns, to, from);

    std::vector<double> rpy;
    std::vector<double> t;
    rpy.reserve(3);
    t.reserve(3);
    bool rotation_exists = node_->get_parameter(rotation_name, rpy);
    bool translation_exists = node_->get_parameter(translation_name, t);

    if (!rotation_exists || !translation_exists) {
        throw std::runtime_error("Rotation or translation not defined: " + translation_name);
    }

    if (rpy.size() != 3 || t.size() != 3) {
        throw std::runtime_error("size mismatch: " + translation_name);
    }

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

    auto T = sf::Transform{sf_quat, {t[0], t[1], t[2]}};

    return T;
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

template <size_t N>
std::array<double, N> ConfigLoader::get_array(const std::string& ns, const std::string& name) {
    assert(std::isalnum(ns.back()));

    std::string qualified_name = ns + "." + name;
    std::vector<double> v;
    v.reserve(N);

    bool exists = node_->get_parameter(qualified_name, v);

    if (!exists) {
        throw std::runtime_error("Parameter does not exist: " + qualified_name);
    }
    if (v.size() != N) {
        throw std::runtime_error("Size mismatch: " + qualified_name);
    }

    auto a = std::array<double, N>{};
    std::copy(v.begin(), v.end(), a.begin());

    return a;
}

sf::Matrix3 ConfigLoader::get_matrix3(const std::string& ns, const std::string& name) {
    auto a = get_array<9>(ns, name);
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
    auto a = get_array<3>(ns, name);
    auto v = sf::Vector3{a[0], a[1], a[2]};
    return v;
}

double ConfigLoader::get_scalar(const std::string& ns, const std::string& name) {
    assert(std::isalnum(ns.back()));
    std::string qualified_name = ns + "." + name;
    double x;
    bool exists = node_->get_parameter(qualified_name, x);
    if (!exists) {
        throw std::runtime_error("Parameter does not exist: " + qualified_name);
    }
    return x;
}
