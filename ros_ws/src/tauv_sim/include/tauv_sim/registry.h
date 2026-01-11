#pragma once

#include <StonefishCommon.h>

struct Material {
    std::string name;
    float density;
    float restitution;
};

struct Look {
    std::string name;
    sf::Color color;
    float roughness;
    float metalness;
    float reflectivity;
};

namespace materials {

static const auto ALUMINUM = Material{"aluminum", 2700.0, 0.8};

static const auto POOL = Material{"pool", 1000.0, 0.5};

static const auto PLASTIC = Material{"plastic", 2000.0, 0.6};

static const auto all_materials = std::array{
    ALUMINUM,
    POOL,
    PLASTIC,
};

}  // namespace materials

namespace looks {

static const auto OSPREY_RED_HULL =
    Look{"osprey_red_hull", sf::Color::RGB(1.0F, 0.0F, 0.0F), 0.8F, 0.8F, 0.1F};

static const auto OSPREY_BLUE_PROP = Look{
    "osprey_blue_prop",
    sf::Color::RGB(0.0F, 0.0F, 0.5F),
    0.3F,
    0.0F,
    0.1F,
};

static const auto all_looks = std::array{
    OSPREY_RED_HULL,
    OSPREY_BLUE_PROP,
};

}  // namespace looks

namespace links {

using Link = std::string;

const Link OSPREY_BASE = "osprey_base";

const Link OSPREY_PRESSURE = "osprey_pressure";

}  // namespace links
