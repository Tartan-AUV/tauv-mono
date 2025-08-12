from numpy._typing import NDArray
from spatialmath import SE3, SO3, UnitQuaternion
import numpy as np
from typing import Union, overload

from geometry_msgs.msg import Transform, Vector3, Wrench, Quaternion, Twist, Pose


def _validate_finite_values(values: Union[list, tuple, np.ndarray], name: str) -> None:
    """Validate that all values are finite (non-nan and non-inf).
    
    Args:
        values: Sequence of numeric values to validate
        name: Descriptive name for error messages
        
    Raises:
        ValueError: If any value is nan or inf
    """
    if isinstance(values, (list, tuple)):
        values = np.array(values)
    
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(values)):
        raise ValueError(f"{name} contains infinite values")


def _validate_vector3(vec: Vector3, name: str) -> None:
    """Validate Vector3 message fields."""
    _validate_finite_values([vec.x, vec.y, vec.z], f"{name} Vector3")


def _validate_quaternion(quat: Quaternion, name: str) -> None:
    """Validate Quaternion message fields."""
    _validate_finite_values([quat.w, quat.x, quat.y, quat.z], f"{name} Quaternion")

@overload
def numpify(obj: Vector3) -> NDArray: ...

@overload
def numpify(obj: Quaternion) -> UnitQuaternion: ...

@overload
def numpify(obj: Pose) -> SE3: ...

@overload
def numpify(obj: Transform) -> SE3: ...

@overload
def numpify(obj: Wrench) -> np.ndarray: ...

@overload
def numpify(obj: Twist) -> NDArray: ...

def numpify(obj: Union[Vector3, Quaternion, Transform, Wrench, Twist]) -> Union[NDArray, UnitQuaternion, SE3, np.ndarray]:
    """Convert ROS geometry messages to numpy/spatialmath objects.
    
    Validates that all input values are finite (non-nan and non-inf).
    
    Raises:
        ValueError: If any input values are nan or inf
        TypeError: If the input type is not supported
    """
    if isinstance(obj, Vector3):
        _validate_vector3(obj, "Vector3")
        return np.array([obj.x, obj.y, obj.z])[:, np.newaxis]
    elif isinstance(obj, Quaternion):
        _validate_quaternion(obj, "Quaternion")
        return UnitQuaternion(obj.w, (obj.x, obj.y, obj.z))
    elif isinstance(obj, Pose):
        # Convert Pose → SE3
        _validate_vector3(obj.position, "Pose position")
        _validate_quaternion(obj.orientation, "Pose orientation")
        q = UnitQuaternion(obj.orientation.w,
                           (obj.orientation.x, obj.orientation.y, obj.orientation.z))
        return SE3.Rt(q.R, (obj.position.x, obj.position.y, obj.position.z))
    elif isinstance(obj, Transform):
        _validate_vector3(obj.translation, "Transform translation")
        _validate_quaternion(obj.rotation, "Transform rotation")
        q = numpify(obj.rotation)
        assert isinstance(q, UnitQuaternion)
        return SE3.Rt(q.R, (obj.translation.x, obj.translation.y, obj.translation.z))
    elif isinstance(obj, Wrench):
        _validate_vector3(obj.force, "Wrench force")
        _validate_vector3(obj.torque, "Wrench torque")
        return np.vstack([numpify(obj.force), numpify(obj.torque)])
    elif isinstance(obj, Twist):
        _validate_vector3(obj.linear, "Twist linear")
        _validate_vector3(obj.angular, "Twist angular")
        return np.vstack([numpify(obj.linear), numpify(obj.angular)])
    else:
        raise TypeError(f"Unsupported type for numpify: {type(obj)}")

def numpify_cov_6x6(covariance: list) -> NDArray:
    """Convert ROS2 covariance array to 6x6 numpy array.
    
    Validates that all input values are finite (non-nan and non-inf).
    
    Args:
        covariance: 36-element list representing row-major flattened 6x6 covariance matrix
        
    Returns:
        6x6 NDArray of float64
        
    Raises:
        ValueError: If any covariance values are nan or inf, or if list length is not 36
    """
    if len(covariance) != 36:
        raise ValueError(f"Covariance list must have 36 elements, got {len(covariance)}")
    
    _validate_finite_values(covariance, "covariance matrix")
    return np.array(covariance, dtype=np.float64).reshape(6, 6)


@overload
def msgify(obj: NDArray, *, message_type: str = None) -> Union[Vector3, Quaternion, Wrench, Twist]: ...

@overload
def msgify(obj: UnitQuaternion) -> Quaternion: ...

@overload
def msgify(obj: SE3) -> Transform: ...

@overload
def msgify(obj: SO3) -> Quaternion: ...

def msgify(obj: Union[NDArray, UnitQuaternion, SE3, SO3, np.ndarray], *, message_type: str = None) -> Union[Vector3, Quaternion, Transform, Wrench, Twist]:
    """Convert numpy/spatialmath objects to ROS geometry messages.
    
    Validates that all input values are finite (non-nan and non-inf).
    
    Args:
        obj: The object to convert
        message_type: For NDArray inputs, specify the target message type ("Vector3", "Quaternion", "Wrench", "Twist").
                     None is allowed for 3-element arrays (defaults to Vector3).
                     Required for other array shapes.
                     
    Raises:
        ValueError: If any input values are nan or inf, or for invalid message types/shapes
        TypeError: If the input type is not supported
    """
    if isinstance(obj, np.ndarray):
        # Validate the entire array first
        _validate_finite_values(obj, "numpy array")
        
        if obj.shape == (3, 1) or (obj.ndim == 1 and obj.shape[0] == 3):
            # 3-element array - default to Vector3 if no message_type specified
            if message_type is None:
                message_type = "Vector3"
            
            vec = obj.flatten()
            if message_type == "Vector3":
                return Vector3(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))
            else:
                raise ValueError(f"Unsupported message_type '{message_type}' for 3-element array")
                
        elif obj.shape == (6, 1) or (obj.ndim == 1 and obj.shape[0] == 6):
            # 6-element array - require message_type specification
            if message_type is None:
                raise ValueError("message_type must be specified for 6-element arrays")
            
            vec = obj.flatten()
            if message_type == "Wrench":
                force = Vector3(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))
                torque = Vector3(x=float(vec[3]), y=float(vec[4]), z=float(vec[5]))
                return Wrench(force=force, torque=torque)
            elif message_type == "Twist":
                linear = Vector3(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))
                angular = Vector3(x=float(vec[3]), y=float(vec[4]), z=float(vec[5]))
                return Twist(linear=linear, angular=angular)
            else:
                raise ValueError(f"Unsupported message_type '{message_type}' for 6-element array")
        else:
            raise ValueError(f"Unsupported numpy array shape for msgify: {obj.shape}")
            
    elif isinstance(obj, UnitQuaternion):
        # Validate quaternion components
        quat_values = [obj.s, obj.vec[0], obj.vec[1], obj.vec[2]]
        _validate_finite_values(quat_values, "UnitQuaternion")
        return Quaternion(x=float(obj.vec[0]), y=float(obj.vec[1]), z=float(obj.vec[2]), w=float(obj.s))
    elif isinstance(obj, SE3):
        # Extract rotation and translation
        rotation_quat = obj.UnitQuaternion()
        translation = obj.t

        # Validate translation vector
        _validate_finite_values(translation, "SE3 translation")

        if message_type == "Pose":
            return Pose(
                position=Vector3(x=float(translation[0]), y=float(translation[1]), z=float(translation[2])),
                orientation=msgify(rotation_quat)
            )
        # Default → Transform
        return Transform(
            translation=Vector3(x=float(translation[0]), y=float(translation[1]), z=float(translation[2])),
            rotation=msgify(rotation_quat)
        )
    elif isinstance(obj, SO3):
        return msgify(obj.UnitQuaternion())
    else:
        raise TypeError(f"Unsupported type for msgify: {type(obj)}")


