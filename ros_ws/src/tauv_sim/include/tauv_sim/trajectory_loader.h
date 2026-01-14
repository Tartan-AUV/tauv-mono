#pragma once

#include <StonefishCommon.h>
#include <entities/animation/PWLTrajectory.h>

#include <string>
#include <vector>

namespace trajectory {

struct Spec {
    sf::PlaybackMode playback_mode;
    std::vector<sf::KeyPoint> keypoints;
};

Spec load_from_yaml(const std::string& path);

}  // namespace trajectory
