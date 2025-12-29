#pragma once

#include <Stonefish/StonefishCommon.h>

#include <string_view>

namespace config {

namespace world {}

namespace osprey {

struct Frames {
    static constexpr std::string_view NS = "osprey.frames";

    // Transform from the CAD frame into the body frame
    sf::Transform cad_T_body;

    // Depth sensor in the body frame
    sf::Vector3 t_depth_B;
};

struct InertialBuoyancy {
    static constexpr std::string_view NS = "osprey.inertial_buoyancy";

    // Mass of the hull in kg
    double mass;

    // Center of mass of the hull expressed in the CAD frame
    sf::Vector3 t_hull_com_C;

    // Inertia tensor of the hull at the center of mass with axes aligned with the CAD frame
    sf::Matrix3 hull_inertia_COM_C;

    // Center of buoyancy of the hull expressed in the CAD frame
    sf::Vector3 t_hull_cob_C;
};

namespace sensors {

struct Depth {
    static constexpr std::string_view NS = "osprey.sensors.depth";

    double noise_std;
    double update_rate;
};

}  // namespace sensors

}  // namespace osprey

}  // namespace config
