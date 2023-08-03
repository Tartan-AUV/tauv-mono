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
    frequency: int
    n_obs: int
    A: np.ndarray = field(default=np.zeros(shape=(3, 3)))
    b: np.ndarray = field(default=np.zeros(shape=(3,)))
    center: np.ndarray = field(default=np.zeros(shape=(3,)))

@dataclass
class Line:
    point: np.ndarray
    unit: np.ndarray

class PingerCluster:
    def __init__(self, min_num_samples=2) -> None:
        self._n_samples = 0
        self.center = None

        self._A = np.zeros(shape=(3, 3))
        self._b = np.zeros(shape=(3,))
        
        self._min_num_samples = min_num_samples
    
    def init_from_lines(self, lines: List[Line], outlier_rejection_thresh=0.9):
        for line in lines:
            self.handle_line(line, outlier_rejection_thresh)

    def handle_line(self, line: Line, outlier_rejection_thresh=0.9):
        def vector_match(v1, v2, sim_thresh=0.9):
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)

            cos_sim = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)

            angular_similarity = 1 - np.arccos(cos_sim) / np.pi

            return angular_similarity >= sim_thresh

        if outlier_rejection_thresh and self.center:
            # get vector from line position to cluster center
            vehicle_to_center = self.center - line.point
            vehicle_to_center /= np.linalg.norm(vehicle_to_center)

            # compare observation unit vector to expected vector
            vm = vector_match(line.unit, vehicle_to_center)
            if vm < outlier_rejection_thresh:
                rospy.loginfo(
                    f"Detected outlier in handle_line... vector match between line and center: {vm}"
                )
                return None

        self._n_samples += 1
        
        point = line.point
        unit = line.unit.reshape((-1, 1))
            
        I = np.eye(3)

        A = (I - unit @ unit.T)
        b = A @ point

        self._A += A
        self._b += b

        if self._n_samples >= self._min_num_samples:
            skew_intersection = np.linalg.pinv(self._A) @ self._b
            est_center = skew_intersection.flatten()

            self.center = est_center
            rospy.loginfo("Estimated pinger center:", self.center)

            return self.center
        else:
            rospy.loginfo("Not enough pinger samples yet")
            return None


class PingerClusterManager:
    def __init__(
        self,
        resample_thresh=4,
        outlier_similarity_thresh=0.9,
        resample_outlier_n_thresh=5,
        obs_queue_len=10,
        min_num_samples=2
    ) -> None:
        self._resample_thresh = resample_thresh
        self._outlier_similarity_thresh = outlier_similarity_thresh
        self._resample_outlier_n_thresh = resample_outlier_n_thresh
        self._obs_queue_len = obs_queue_len
        self._min_num_samples = min_num_samples

        self._outlier_queue: List[Line] = []
        self._observation_result_queue = []

        self._pc = PingerCluster(self._min_num_samples)

    def handle_line(self, line: Line):
        handle_result = self._pc.handle_line(
            line,
            outlier_rejection_thresh=self._outlier_similarity_thresh
        )

        if self._pc.center is not None:
            if handle_result is None:
                self._observation_result_queue.append(1)
                self._outlier_queue.append(line)
            else:
                self._observation_result_queue.append(0)
        
            # maintain fixed length queue
            if len(self._observation_result_queue) >= self._obs_queue_len:
                self._observation_result_queue.pop(0)

            # check if we have enough outliers to redo cluster
            if sum(self._observation_result_queue) >= self._resample_outlier_n_thresh:
                rospy.loginfo("Detected a pinger location switch, rebuilding cluster...")
                
                # redo the pinger cluster
                pc = PingerCluster(min_num_samples=self._min_num_samples)
                pc.init_from_lines(
                    self._outlier_queue,
                    self._outlier_similarity_thresh
                )

                self._observation_result_queue = []
                self._outlier_queue = []

                self._pc = pc
            
        return self._pc.center


class PingerRayIntersection:
    def __init__(self) -> None:
        self._A = np.zeros(shape=(3, 3))
        self._b = np.zeros(shape=(3,))
        
        self._direction_sub = rospy.Subscriber(f'vehicle/pinger_localizer/direction', Vector3Stamped, self._update_fit)
        self._pinger_loc_pub = rospy.Publisher(f'vehicle/pinger_localizer/pinger_ray', PointStamped, queue_size=10)

        self._cluster_manager = PingerClusterManager()

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

        line = Line(
            point=vehicle_pos,
            unit=world_direction_vector
        )

        est_center = self._cluster_manager.handle_line(line)

        if est_center is not None:
            rospy.loginfo("Got pinger location estimate:", est_center)
            
            pinger_pos_msg = PointStamped(
                header=Header(
                    stamp=rospy.Time.now(),
                    frame_id="kf/odom"
                ),
                point=tm(est_center, Point)
            )

            self._pinger_loc_pub.publish(pinger_pos_msg)