import rospy
import serial
import numpy as np
from math import atan2

class ModemLocalizer:

    def __init__(self):
        self._load_config()

        self._stamps = [0, 0, 0, 0]

    def start(self):
        while not self._open() and not rospy.is_shutdown():
            rospy.logerr('Open failed, trying again.')
            rospy.sleep(1.0)

        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _open(self):
        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=1.0, writeTimeout=1.0)
        except serial.SerialException as e:
            rospy.logerr(f'Open failed: {e}')
            return False

        return True

    def _update(self, timer_event):
        while self._serial.in_waiting() > 0:
            line = self._serial.read_until(expected='\r\n', size=10)

            rospy.loginfo(f'Got line: {line.hex()}')

            if len(line) < 10:
                break

            word = int(line.decode("utf-8")[0:8], 16)

            channel = word & 0xC0000000
            stamp = word & 0x3FFFFFFF

            self._handle_edge(channel, stamp)

        self._serial.reset_input_buffer()

    def _handle_edge(self, channel, stamp):
        rospy.loginfo(f'Got edge: {channel}, {stamp}')

        # Reset on overflow
        if stamp < max(self._stamps):
            self._stamps = [0, 0, 0, 0]
            return

        self._stamps[channel] = stamp

        # Flush old edges
        for i in range(4):
            if (max(self._stamps) - self._stamps[i]) / self._stamp_frequency > self._max_delay:
                self._stamps[i] = 0

        # Check for complete detection
        if 0 not in self._stamps:
            delays = [(stamp - min(self._stamps)) / self._stamp_frequency for stamp in self._stamps]

            rospy.loginfo(f'Got delays: {delays}')

            # Find direction
            A = np.array([
                self._positions[1] - self._positions[0],
                self._positions[2] - self._positions[0],
                self._positions[3] - self._positions[0],
            ])

            b = np.array([
                delays[1] - delays[0],
                delays[2] - delays[0],
                delays[3] - delays[0]
            ])

            x, r = np.linalg.lstsq(A, b, rcond=None)[0]

            x_norm = x / np.linalg.norm(x)

            rospy.loginfo(f'Got direction: {x_norm}, {r}')

            heading = atan2(x_norm[1], x_norm[0])

            rospy.loginfo(f'Got heading: {heading}')

            self._stamps = [0, 0, 0, 0]

    def _load_config(self):
        self._dt = 1.0 / rospy.get_param('~update_frequency')
        self._port = rospy.get_param('~port')
        self._baud = rospy.get_param('~baud')
        self._max_delay = rospy.get_param('~max_delay')
        self._stamp_frequency = rospy.get_param('~stamp_frequency')
        self._positions = np.array(rospy.get_param('~positions'))


def main():
    rospy.init_node('modem_localizer')
    m = ModemLocalizer()
    m.start()