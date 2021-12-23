import rospy
from typing import List

from tauv_msgs.msg import ImuSync as ImuSyncMsg, DvlData as DvlDataMsg

from .pathfinder import Pathfinder


class TeledyneDVL:
    def __init__(self):
        self._data_pub: rospy.Publisher = rospy.Publisher('/teledyne_dvl/data', DvlDataMsg, queue_size=10)
        self._sync_sub: rospy.Subscriber = rospy.Subscriber('/xsens_imu/sync', ImuSyncMsg, self._handle_sync)

        self._sync_timestamps: List[rospy.Time] = []

        port = rospy.get_param('~port')
        baudrate = rospy.get_param('~baudrate')

        self._pf: Pathfinder = Pathfinder(port, baudrate)

    def start(self):
        self._pf.open()

        while not rospy.is_shutdown():
            ensemble = self._pf.poll()

            if ensemble is None:
                print('No ensemble')
                continue

            self._sweep_sync_timestamps(ensemble.receive_time)

            msg: DvlDataMsg = ensemble.to_msg()

            if len(self._sync_timestamps) == 0:
                print('No sync timestamps')
                continue

            msg.header.stamp = self._sync_timestamps[0] + rospy.Duration(Pathfinder.TOV_TIME)
            self._sync_timestamps = self._sync_timestamps[1:]

            self._data_pub.publish(msg)

        self._pf.close()

    def _handle_sync(self, data: ImuSyncMsg):
        time: rospy.Time = data.header.stamp

        self._sync_timestamps.append(time)

    def _sweep_sync_timestamps(self, time: rospy.Time):
        self._sync_timestamps = list(filter(
                lambda t: rospy.Duration(Pathfinder.MIN_MEASURE_TIME) < time - t < rospy.Duration(Pathfinder.MAX_MEASURE_TIME),
                self._sync_timestamps
        ))


def main():
    rospy.init_node('teledyne_dvl')
    n = TeledyneDVL()
    n.start()
