from numpy._typing import NDArray
from spatialmath import SE3, SO3, UnitQuaternion
import numpy as np
from typing import overload

from geometry_msgs.msg import Transform, Vector3, Wrench, Quaternion

@overload
def numpify(v: Vector3) -> NDArray:
    return np.array([v.x, v.y, v.z])

@overload
def numpify(q: Quaternion) -> UnitQuaternion:
    return UnitQuaternion(q.w, (q.x, q.y, q.z))

@overload
def numpify(T: Transform) -> SE3:
    q = numpify(T.rotation)
    return SE3.Rt(q.SO3(), (T.translation.x, T.translation.y, T.translation.z))

def numpify_cov()

def numpify(F: Wrench) -> np.ndarray:
    return np.hstack([numpify(F.force), numpify(F.torque)])
