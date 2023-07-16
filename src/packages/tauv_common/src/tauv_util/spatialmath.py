from spatialmath.base.types import R3
from spatialmath import SO3, SE3, UnitQuaternion

from geometry_msgs.msg import Vector3 as Vector3Msg, Point as PointMsg, Quaternion as QuaternionMsg, Transform as TransformMsg


def r3_to_ros_vector3(x: R3) -> Vector3Msg:
    return Vector3Msg(x[0], x[1], x[2])


def ros_vector3_to_r3(x: Vector3Msg) -> R3:
    return R3([x.x, x.y, x.z])


def r3_to_ros_point(x: R3) -> PointMsg:
    return PointMsg(x[0], x[1], x[2])


def ros_point_to_r3(x: PointMsg) -> R3:
    return R3([x.x, x.y, x.z])


def ros_quaternion_to_unit_quaternion(x: QuaternionMsg) -> UnitQuaternion:
    return UnitQuaternion(s=x.w, v=R3([x.x, x.y, x.z]))


def unit_quaternion_to_ros_quaternion(x: UnitQuaternion) -> QuaternionMsg:
    return QuaternionMsg(w=x.s, x=x.v.x, y=x.v.y, z=x.v.z)


def ros_transform_to_se3(x: TransformMsg) -> SE3:
    return SE3.Rt(ros_quaternion_to_unit_quaternion(x.rotation).SO3(), ros_vector3_to_r3(x.translation))


def se3_to_ros_transform(x: SE3) -> TransformMsg:
    return TransformMsg(
        translation=r3_to_ros_vector3(x.t),
        rotation=unit_quaternion_to_ros_quaternion(UnitQuaternion(x)),
    )
