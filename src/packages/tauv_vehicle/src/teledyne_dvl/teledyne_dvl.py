import rospy

from tauv_msgs.msg import ImuSync as ImuSyncMsg, DvlData as DvlDataMsg

from pathfinder import Pathfinder

class TeledyneDVL:
    def __init__(self):
        self.data_pub = rospy.Publisher('/sensors/dvl/data', DvlDataMsg, queue_size=10)
        self.sync_sub = rospy.Subscriber('/sensors/imu/sync', ImuSyncMsg, self._handle_imu_sync)

        self.sync_msgs = []

        port = rospy.get_param('port')
        baudrate = rospy.get_param('baudrate')

        self.pf = Pathfinder(port, baudrate)

    def start(self):
        self.pf.open()
        rospy.spin()

    def _handle_imu_sync(self, msg):
        if not msg.triggered_dvl:
            return

        self._poll()

        self.sync_msgs.append(msg)

    def _poll(self):
        try:
            ensemble = self.pf.poll()

            # TODO: Pair ensemble with a sync message

            self.data_pub.publish(ensemble.to_msg())
        except Exception as e:
            rospy.logerr(e)

def main():
    rospy.init_node('teledyne_dvl')
    n = TeledyneDVL()
    n.start()
