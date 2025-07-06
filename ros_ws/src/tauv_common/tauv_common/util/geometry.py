from spatialmath import SE3, SO3, UnitQuaternion
import numpy as np

from geometry_msgs.msg import Transform, Vector3, Wrench

def tf2_transform_to_SE3(tf: Transform) -> SE3:
    q = UnitQuaternion(tf.rotation.w, (tf.rotation.x, tf.rotation.y, tf.rotation.z))
    return SE3.Rt(q.SO3(), (tf.translation.x, tf.translation.y, tf.translation.z))

def vector3_msg_to_numpy(vector3: Vector3) -> np.ndarray:
    return np.array([vector3.x, vector3.y, vector3.z])

def wrench_msg_to_numpy(wrench: Wrench) -> np.ndarray:
    return np.hstack((vector3_msg_to_numpy(wrench.force), vector3_msg_to_numpy(wrench.torque)))
