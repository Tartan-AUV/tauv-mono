import rospy
import serial
import bitstring

from teledyne_dvl.ensemble import Ensemble

class Pathfinder:

    TIMEOUT = 1.0

    def __init__(self, port: str, baudrate: int):
        self._conn = serial.Serial(port=port, baudrate=baudrate, timeout=Pathfinder.TIMEOUT)
        self._measuring = False

    def open(self):
        self._log('open')

        if not self._conn.isOpen():
            self._conn.open()

        self._send_command('===')

        rospy.sleep(rospy.Duration(1.0))

        self._configure()

        rospy.sleep(rospy.Duration(1.0))

        self.start_measuring()

        rospy.sleep(rospy.Duration(1.0))

    def close(self):
        self._log('close')

        self.stop_measuring()

        self._conn.close()

    def start_measuring(self):
        self._log('start_measuring')

        self._send_command('CS')

        self._measuring = True

    def stop_measuring(self):
        self._log('stop_measuring')

        self._send_break()

        self._measuring = False

    def poll(self) -> Ensemble:
        self._log('poll')

        timeout = rospy.Time.now() + rospy.Duration(1.0)

        header_id = self._read(1)
        while header_id is None or header_id.hex() != '7f':
            self._log('header_id', header_id.hex())

            if rospy.Time.now() > timeout:
                self._log('timeout!')
                return None

            header_id = self._read(1)

        self._read(1)
        e = Ensemble()

        header_data = self._read(Ensemble.HEADER_SIZE - Ensemble.ID_SIZE)
        self._log('header_data', header_data.hex())

        header = bitstring.BitStream(bytes=header_id + header_id + header_data)
        
        packet_size = e.parse_header(header)

        packet_data = self._read(packet_size)
        self._log('packet_data', packet_data.hex())

        packet = bitstring.BitStream(bytes=packet_data)

        e.parse_packet(packet)

        return e

    def _write(self, data: bytes):
        self._log('[write]', data.hex())
        self._conn.write(data)
        self._conn.flush()

    def _read(self, size: int) -> bytes:
        data = self._conn.read(size)
        self._log('[read]', data.hex())
        return data

    def _send_break(self):
        self._log('[break]')
        self._conn.send_break(duration=0.3)

    def _send_command(self, cmd: str):
        self._log('[cmd]', cmd)
        cmd += '\r'
        data = cmd.encode('ascii')
        self._write(data)

    def _configure(self):
        self._log('_configure')

        self._send_command('PD0')

    def _log(self, *args):
        print(args)
