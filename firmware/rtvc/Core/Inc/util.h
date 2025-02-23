//
// Created by gleb on 4/11/25.
//

#ifndef UTIL_H
#define UTIL_H

#define copy_vec3(src, dest) { (dest).x = (src).x; (dest).y = (src).y; (dest).z = (src).z; }
#define copy_quat(src, dest) { (dest).w = (src).w; (dest).x = (src).x; \
                               (dest).y = (src).y; (dest).z = (src).z; }

typedef struct {
    float x, y, z;
} Vec3_t;

typedef struct {
    float w, x, y, z;
} Quat_t;

#endif //UTIL_H
