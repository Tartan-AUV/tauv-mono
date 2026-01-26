import csv
import math
import os

from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

bag_name = "sim_2026.01.26_03.09.26"

bag_path = "./rosbags/" + bag_name
csv_path = "./formatted_csvs/" + bag_name + ".csv"

# Open bag
reader = SequentialReader()
storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
converter_options = ConverterOptions('', '')
reader.open(storage_options, converter_options)


def quaternion_to_rpy(q):
    """Convert quaternion to roll, pitch, yaw"""
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    # CSV header
    header = [
        "time",
        # Position
        "x",
        "y",
        "z",
        # Orientation (quat)
        "qx",
        "qy",
        "qz",
        "qw",
        # Orientation (RPY)
        "roll",
        "pitch",
        "yaw",
        # Linear velocity
        "vx",
        "vy",
        "vz",
        # Angular velocity
        "wx",
        "wy",
        "wz",
    ]

    # Pose covariance (36)
    header += [f"pose_cov_{i}" for i in range(36)]

    # Twist covariance (36)
    header += [f"twist_cov_{i}" for i in range(36)]

    writer.writerow(header)

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != "/odometry/filtered":
            continue

        msg = deserialize_message(data, Odometry)

        # Time (seconds)
        time_sec = t * 1e-9

        # Pose
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        roll, pitch, yaw = quaternion_to_rpy(q)

        # Twist
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        row = [
            time_sec,
            p.x,
            p.y,
            p.z,
            q.x,
            q.y,
            q.z,
            q.w,
            roll,
            pitch,
            yaw,
            v.x,
            v.y,
            v.z,
            w.x,
            w.y,
            w.z,
        ]

        # Covariances
        row += list(msg.pose.covariance)
        row += list(msg.twist.covariance)

        writer.writerow(row)

print(f"CSV saved to {os.path.abspath(csv_path)}")
