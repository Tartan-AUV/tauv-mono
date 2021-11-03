import serial
import bitstring

from ensemble import Ensemble

class Pathfinder:

    TIMEOUT = 100

    def __init__(self, port, baudrate):
        self._conn = serial.Serial(port=port, baudrate=baudrate, timeout=Pathfinder.TIMEOUT)
        self._measuring = False

    def open(self):
        if not self._conn.isOpen():
            self._conn.open()

        self._conn.send_break()
        self._conn.send_break()

        self._conn.flush()

        self._configure()

        self.start_measuring()

    def close(self):
        self.stop_measuring()
        self._conn.close()

    def start_measuring():
        self._conn.write('CS\n')

        self._measuring = True

    def stop_measuring():
        self._conn.send_break()

        self._measuring = False

    def poll(self):
        if not self.measuring:
            return

        try:
            header_id = self._conn.read(size=Ensemble.ID_SIZE).hex()
            if not header_id == Ensemble.HEADER_ID:
                self._conn.flush()
                raise ValueError('Unexpected Header ID: {}'.format(header_id))

            e = Ensemble()

            header_data = self._conn.read(size=Ensemble.HEADER_SIZE - Ensemble.ID_SIZE).hex()
            header = bitstring.BitStream('0x:{}{}'.format(header_id, header_data))
            
            packet_size = e.parse_header(header)

            packet_data = self._conn.read(size=packet_size).hex()
            packet = bitstring.BitStream('0x:{}'.format(packet_data))

            e.parse_packet(packet)

            return e
            
        except Exception as e:
            rospy.logerr(e)

    def _configure(self):
        # TODO: Add code to configure Pathfinder
