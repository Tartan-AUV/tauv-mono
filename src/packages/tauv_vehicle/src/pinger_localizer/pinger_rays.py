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


class RayIntersection:
    def __init__(self) -> None:
        self._A = np.zeros(shape=(3, 3))
        self._b = np.zeros(shape=(3,))
        self._n_samples = 0
    
    def indv_fit(self, lines, use_old_pts=True):
        self._n_samples += 1

        A = np.zeros(shape=(3, 3))
        b = np.zeros(shape=(3,))
        
        for line in lines:
            point, unit = line
            unit = unit.reshape((-1, 1))
            
            I = np.eye(3)
            M = (I - unit @ unit.T)
            
            A += M
            b += M @ point

        if use_old_pts:
            full_A = self._A + A
            full_b = self._b + b

            self._A += A
            self._b += b
        else:
            full_A = A
            full_b = b

        skew_intersection = np.linalg.pinv(full_A) @ full_b
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

        pinger_pos_msg = PointStamped(
            header=Header(
                stamp=rospy.Time.now(),
                frame_id="kf/odom"
            ),
            point=tm(pinger_pos, Point)
        )

        self._pinger_loc_pub.publish(pinger_pos_msg)