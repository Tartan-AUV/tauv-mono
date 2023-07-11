from spatialmath.base.types import R3

from geometry_msgs.msg import Vector3, Point

def r3_to_vector3(x: R3) -> Vector3:
    return Vector3(x.x, x.y, x.z)

def vector3_to_r3(x: Vector3) -> R3:
    return R3(x.x, x.y, x.z)

def r3_to_point(x: R3) -> Point:
    return Point(x.x, x.y, x.z)

def point_to_r3(x: Point) -> R3:
    return R3(x.x, x.y, x.z)

