import rospy
from typing import List

from tauv_msgs.msg import XsensImuSync as ImuSyncMsg, TeledyneDvlData as DvlDataMsg

from .pathfinder import Pathfinder


class TeledyneDVL:
    def __init__(self):
        self._data_pub: rospy.Publisher = rospy.Publisher('dvl_data', DvlDataMsg, queue_size=10)
        self._sync_sub: rospy.Subscriber = rospy.Subscriber('imu_sync', ImuSyncMsg, self._handle_sync)

        self._sync_timestamps: List[rospy.Time] = []

        port = rospy.get_param('~port')
        baudrate = rospy.get_param('~baudrate')

        self._pf: Pathfinder = Pathfinder(port, baudrate)

    def start(self):
        self._pf.open()

        while not rospy.is_shutdown():
            ensemble = self._pf.poll()

            if ensemble is None:
                rospy.logwarn('No ensemble')
                continue

            self._sweep_sync_timestamps(ensemble.receive_time)

            msg: DvlDataMsg = ensemble.to_msg()

            if len(self._sync_timestamps) > 0:
                msg.header.stamp = self._sync_timestamps[0] + rospy.Duration.from_sec(sum(msg.beam_time_to_bottoms) / 4)
                self._sync_timestamps = self._sync_timestamps[1:]
            else:
                rospy.logwarn('No sync timestamps')

            self._data_pub.publish(msg)

        self._pf.close()

    def _handle_sync(self, data: ImuSyncMsg):
        time: rospy.Time = data.header.stamp

        if data.triggered_dvl:
            self._sync_timestamps.append(time)

    def _sweep_sync_timestamps(self, time: rospy.Time):
        print(map(lambda t: time - t, self._sync_timestamps))
        self._sync_timestamps = list(filter(
                lambda t: rospy.Duration(Pathfinder.MIN_MEASURE_TIME) < time - t < rospy.Duration(Pathfinder.MAX_MEASURE_TIME),
                self._sync_timestamps
        ))


def main():
    rospy.init_node('teledyne_dvl')
    n = TeledyneDVL()
    n.start()
