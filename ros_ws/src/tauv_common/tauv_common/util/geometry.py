from numpy._typing import NDArray
from spatialmath import SE3, SO3, UnitQuaternion
import numpy as np
from typing import Union

from geometry_msgs.msg import Transform, Vector3, Wrench, Quaternion

def numpify(obj: Union[Vector3, Quaternion, Transform, Wrench]) -> Union[NDArray, UnitQuaternion, SE3, np.ndarray]:
    """Convert ROS geometry messages to numpy/spatialmath objects."""
    if isinstance(obj, Vector3):
        return np.array([obj.x, obj.y, obj.z])[:, np.newaxis]
    elif isinstance(obj, Quaternion):
        return UnitQuaternion(obj.w, (obj.x, obj.y, obj.z))
    elif isinstance(obj, Transform):
        q = numpify(obj.rotation)
        assert isinstance(q, UnitQuaternion)
        return SE3.Rt(q.R, (obj.translation.x, obj.translation.y, obj.translation.z))
    elif isinstance(obj, Wrench):
        return np.vstack([numpify(obj.force), numpify(obj.torque)])
    else:
        raise TypeError(f"Unsupported type for numpify: {type(obj)}")

def numpify_cov_6x6(covariance: list) -> NDArray:
    """Convert ROS2 covariance array to 6x6 numpy array.
    
    Args:
        covariance: 36-element list representing row-major flattened 6x6 covariance matrix
        
    Returns:
        6x6 NDArray of float64
    """
    return np.array(covariance, dtype=np.float64).reshape(6, 6)

def wrench_msg_to_numpy(wrench: Wrench) -> NDArray:
    """Convert a Wrench message to a 6x1 numpy array [fx, fy, fz, tx, ty, tz].
    
    Args:
        wrench: geometry_msgs/Wrench message
        
    Returns:
        6x1 NDArray with forces and torques
    """
    return numpify(wrench)
