import rospy
from tauv_msgs.msg import ModemFrame

import serial

import time
from collections import deque

_SET_MODE = b'\x00'
_SET_PARAM = b'\x01'
_MASTER_REQ = b'\x02'
_SLAVE_RESP = b'\x03'
_SLAVE_REQ = b'\x04'
_MASTER_RESP = b'\x05'

_MODE_MASTER = b'\x00'
_MODE_SLAVE = b'\x01'

class ModemNode:
    def __init__(self):
        self._port = rospy.get_param('teensy_port')
        self._baud_rate = rospy.get_param('baud_rate', 115200)
        self._queue_size = rospy.get_param('queue_size', 10)
        self._mode = rospy.get_param('mode', 'master')
        if self._mode == 'master':
            callback = self._tx_master_callback
        elif self._mode == 'slave':
            callback = self._tx_slave_callback
        else:
            raise RuntimeError('Incorrect mode')

        self._pub = rospy.Publisher('vehicle/modem/rx', ModemFrame, queue_size = self._queue_size)
        self._sub = rospy.Subscriber('vehicle/modem/tx', ModemFrame, callback)
        self._ser = serial.Serial(self._port, self._baud_rate, timeout = 1)
        self._tx_queue = deque()

    def start(self):
        while not self._ser.is_open and not rospy.is_shutdown():
            print('Cannot open the serial port')
            time.sleep(1.0)

        self._init_teensy()

        if self._mode == 'master':
            self._run_master()
        elif self._mode == 'slave':
            self._run_slave()

    def _init_teensy(self):
        # set mode
        self._ser.write(0x00)

    def _run_master(self):
        rospy.spin()

    def _run_slave(self):
        while not rospy.is_shutdown():
            while not self._ser.in_waiting:
                time.sleep(0.2)

            cmd = self._ser.read()

            if cmd != _SLAVE_REQ:
                continue

            size = int(self._ser.read())
            data = self._ser.read(size)

            if len(self._tx_queue):
                payload = self._tx_queue.pop()
            else:
                payload = b''

            self._ser.write(_SLAVE_RESP)
            assert(len(payload) < 256)
            self._ser.write(bytes(len(payload)))
            self._ser.write(payload)

            frame = ModemFrame()
            # todo: set header
            frame.data = data
            self._pub.publish(frame)

    def _tx_master_callback(self, frame):
        payload = frame.data
        assert(len(payload) < 256)

        self._ser.write(_MASTER_REQ)
        self._ser.write(bytes(len(payload)))
        self._ser.write(payload)

        start_time = time.time()
        while not self._ser.in_waiting and time.time() < start_time + 1.0:
            time.sleep(0.05)

        cmd = self._ser.read()
        if cmd != _MASTER_RESP:
            return

        size = self._ser.read()
        payload = self._ser.read(size)

        frame = ModemFrame()
        frame.data = payload
        self._pub.publish(frame)

    def _tx_slave_callback(self, frame):
        self._tx_queue.appendleft(frame.data)


def main():
    rospy.init_node('modem')
    n = ModemNode()
    n.start()
