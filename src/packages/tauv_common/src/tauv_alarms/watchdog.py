from socket import timeout
import rospy
from tauv_alarms.alarm_client import Alarm, AlarmClient
from sensor_msgs.msg import CameraInfo
from tauv_msgs.msg import Pose

KickableDeadlines = {
    Alarm.CAMERA_NO_VIDEO_FRONT: rospy.Time(0),
    Alarm.CAMERA_NO_VIDEO_BOTTOM: rospy.Time(0),
    Alarm.GNC_NO_POSE: rospy.Time(0)
}

class Watchdog:
    def __init__(self) -> None:
        self.ac = AlarmClient()

        self.cam_frontsub = rospy.Subscriber("/zedm_A/zed_node_A/left/camera_info", CameraInfo, lambda m: self.kick(m, Alarm.CAMERA_NO_VIDEO_FRONT, 1.0))
        self.cam_bottomsub = rospy.Subscriber("/zedm_B/zed_node_B/left/camera_info", CameraInfo, lambda m: self.kick(m, Alarm.CAMERA_NO_VIDEO_BOTTOM, 1.0))
        self.cam_bottomsub = rospy.Subscriber("/gnc/pose", Pose, lambda m: self.kick(m, Alarm.GNC_NO_POSE, 1.0))


    def check(self):
        n = rospy.Time.now()
        for a in KickableDeadlines.keys():
            self.ac.set(a, "set by watchdog", n > KickableDeadlines[a])

    def kick(self, msg, a: Alarm, timeout: float = 1.0):
        KickableDeadlines[a] = rospy.Time.now() + rospy.Duration(timeout)

def main():
    rospy.init_node('watchdog')
    Watchdog()
    rospy.spin()