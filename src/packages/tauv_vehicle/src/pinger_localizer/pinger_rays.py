import numpy as np
import random
import rospy
from nav_msgs.msg import Odometry as OdometryMsg
from geometry_msgs.msg import Vector3Stamped, PointStamped, Point
from std_msgs.msg import Header
import tf2_ros
import tf
import message_filters
from tauv_util.types import tl, tm
from typing import List
from dataclasses import dataclass, field

@dataclass
class PingerCluster:
    id: int
    n_obs: int
    A: np.ndarray = field(default=np.zeros(shape=(3, 3)))
    b: np.ndarray = field(default=np.zeros(shape=(3,)))
    center: np.ndarray = field(default=np.zeros(shape=(3,)))

@dataclass
class Line:
    point: np.ndarray
    unit: np.ndarray

class RayIntersection:
    def __init__(self) -> None:
        # Store 
        self._clusters: List[PingerCluster] = []
        self._unmatched_detections = List[Line] = []

        self._A = np.zeros(shape=(3, 3))
        self._b = np.zeros(shape=(3,))
        self.n_samples = 0
    
    def indv_fit(self, line):
        def vector_match(v1, v2, sim_thresh=0.9):
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            cos_sim = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angular_similarity = 1 - np.arccos(cos_sim) / np.pi

            return angular_similarity >= sim_thresh

        cluster_id = None

        # Determine which cluster, if any, the line points towards
        # First try existing clusters
        for i, cluster in enumerate(self._clusters):
            bearing_vector = cluster.center - point
            bearing_vector /= np.linalg.norm(bearing_vector)

            if vector_match(bearing_vector, unit):
                cluster_id = i
                break

        # Next try unmatched lines
        for unmatched_line in self._unmatched_detections:
            
        
        if cluster_id:
            self._clusters[cluster_id]
        else:
            self._unmatched_detections.append(line)

        # Math taken from:
        #   https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#In_more_than_two_dimensions
        
        self.n_samples += 1

        if not isinstance(lines, list):
            lines = [lines]

        A = np.zeros(shape=(3, 3))
        b = np.zeros(shape=(3,))
        
        for line in lines:
            point, unit = line
            unit = unit.reshape((-1, 1))
            
            I = np.eye(3)
            M = (I - unit @ unit.T)
            
            A += M
            b += M @ point

        full_A = self._A + A
        full_b = self._b + b
        
        # Check the angle between the lines and the intersection point
        skew_intersection = np.linalg.pinv(full_A) @ full_b

        for line in lines:
            point, unit = line
            norm_unit = np.linalg.norm(unit)

            pos_to_intersection = skew_intersection - point
            norm_pos_to_intersection = np.linalg.norm(pos_to_intersection)

            cos_theta = np.dot(norm_unit, pos_to_intersection) / (norm_unit * norm_pos_to_intersection)
            theta = np.arccos(cos_theta)

            if theta > np.pi / 2: 
                raise RuntimeError
        
        self._A = full_A
        self._b = full_b

        return skew_intersection.flatten()


class PingerRayIntersection:
    def __init__(self) -> None:
        self._A = np.zeros(shape=(3, 3))
        self._b = np.zeros(shape=(3,))
        
        self._direction_sub = rospy.Subscriber(f'vehicle/pinger_localizer/direction', Vector3Stamped, self._update_fit)
        self._pinger_loc_pub = rospy.Publisher(f'vehicle/pinger_localizer/pinger_ray', PointStamped, queue_size=10)

        self._ray_intersection = RayIntersection()

    def _update_fit(self, direction: Vector3Stamped):
        direction_time = rospy.Time(
            secs=direction.header.stamp.secs,
            nsecs=direction.header.stamp.nsecs
        )

        world_direction = tf.TransformerROS.transformVector3(target_frame='kf/odom', v3s=direction)
        world_direction_vector = np.array(tl(world_direction.vector))
        world_direction_vector /= np.linalg.norm(world_direction_vector)

        vehicle_pos, _ = tf.Transformer.lookupTransform(target_frame='kf/odom', source_frame=direction.header.frame_id, time=direction_time)
        vehicle_pos = np.array(vehicle_pos)

        line = [
            vehicle_pos,
            world_direction_vector
        ]

        pinger_pos = list(self._ray_intersection.indv_fit(line))
        
        n_samples = self._ray_intersection.n_samples
        if n_samples > 0 and n_samples % 50 == 0:
            rospy.logdebug(f"Pinger position sampled from {n_samples} rays")

        pinger_pos_msg = PointStamped(
            header=Header(
                stamp=rospy.Time.now(),
                frame_id="kf/odom"
            ),
            point=tm(pinger_pos, Point)
        )

        self._pinger_loc_pub.publish(pinger_pos_msg)