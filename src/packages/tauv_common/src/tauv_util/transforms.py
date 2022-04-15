import numpy as np
from math import cos, sin
from geometry_msgs.msg import Pose, Twist, Vector3, Quaternion
from scipy.spatial.transform import Rotation

from .types import tl, tm


def quat_to_rpy(orientation: Quaternion) -> np.array:
    return np.flip(Rotation.from_quat(tl(orientation)).as_euler('ZYX'))


def rpy_to_quat(orientation: np.array) -> Quaternion:
    return tm(Rotation.from_euler('ZYX', np.flip(orientation)).as_quat(), Quaternion)


def linear_body_to_world_matrix(pose: Pose) -> np.array:
    orientation = quat_to_rpy(pose.orientation)

    cr = cos(orientation[0])
    sr = sin(orientation[0])
    cp = cos(orientation[1])
    sp = sin(orientation[1])
    cy = cos(orientation[2])
    sy = sin(orientation[2])

    body_to_world = np.array([
        [cp * cy, sp * cy * sr - sy * cr, sp * cy * cr + sy * sr],
        [cp * sy, sp * sy * sr + cy * cr, sp * sy * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])

    return body_to_world


def angular_body_to_world_matrix(pose: Pose) -> np.array:
    orientation = quat_to_rpy(pose.orientation)

    cr = cos(orientation[0])
    sr = sin(orientation[0])
    cp = cos(orientation[1])
    sp = sin(orientation[1])

    body_to_world = np.array([
        [1, 0, -sp],
        [0, cr, sr * cp],
        [0, -sr, cr * cp]
    ])

    return body_to_world


def twist_body_to_world(pose: Pose, twist: Twist) -> Twist:
    body_velocity = np.array(tl(twist.linear))
    body_angular_velocity = np.array(tl(twist.angular))

    world_velocity = linear_body_to_world_matrix(pose) @ body_velocity

    world_angular_velocity = angular_body_to_world_matrix(pose) @ body_angular_velocity

    world_twist = Twist(
        linear=tm(world_velocity, Vector3),
        angular=tm(world_angular_velocity, Vector3)
    )

    return world_twist


def twist_world_to_body(pose: Pose, twist: Twist) -> Twist:
    world_velocity = np.array(tl(twist.linear))
    world_angular_velocity = np.array(tl(twist.angular))

    body_velocity = np.linalg.inv(linear_body_to_world_matrix(pose)) @ world_velocity
    body_angular_velocity = np.linalg.inv(angular_body_to_world_matrix(pose)) @ world_angular_velocity

    body_twist = Twist(
        linear=tm(body_velocity, Vector3),
        angular=tm(body_angular_velocity, Vector3)
    )

    return body_twist

def linear_distance(a: Pose, b: Pose) -> float:
    return np.linalg.norm(tl(a.position) - tl(b.position))

def yaw_distance(a: Pose, b: Pose) -> float:
    return np.abs(quat_to_rpy(a.orientation)[2] - quat_to_rpy(b.orientation)[2])