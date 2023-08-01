#pragma once
#include <cstdint>
namespace dai {
/**
 * Which Camera socket to use.
 *
 * AUTO denotes that the decision will be made by device
 */
enum class CameraBoardSocket : int32_t {
    AUTO = -1,
    RGB,
    LEFT,
    RIGHT,
    CENTER = RGB,
    CAM_A = RGB,
    CAM_B = LEFT,
    CAM_C = RIGHT,
    CAM_D,
    CAM_E,
    CAM_F,
    CAM_G,
    CAM_H,
};

}  // namespace dai
