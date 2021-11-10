import rospy

from tauv_msgs.msg import ImuSync as ImuSyncMsg, DvlData as DvlDataMsg

from teledyne_dvl.pathfinder import Pathfinder

class TeledyneDVL:
    def __init__(self):
        self.data_pub = rospy.Publisher('/sensors/dvl/data', DvlDataMsg, queue_size=10)
        self.sync_sub = rospy.Subscriber('/sensors/imu/sync', ImuSyncMsg, self._handle_imu_sync)

        self.sync_msgs = []

        port = rospy.get_param('~port')
        baudrate = rospy.get_param('~baudrate')

        self.pf = Pathfinder(port, baudrate)

    def start(self):
        self.pf.open()
        self._run()

    def _handle_imu_sync(self, msg: ImuSyncMsg):
        if not msg.triggered_dvl:
            return

        self._poll()

        self.sync_msgs.append(msg)

    def _run(self):
        while not rospy.is_shutdown():
            ensemble = self.pf.poll()

            print('ensemble', ensemble)

            # TODO: Pair ensemble with a sync message

            if not ensemble is None:
                self.data_pub.publish(ensemble.to_msg())

        self.pf.close()

def main():
    rospy.init_node('teledyne_dvl')
    n = TeledyneDVL()
    n.start()
