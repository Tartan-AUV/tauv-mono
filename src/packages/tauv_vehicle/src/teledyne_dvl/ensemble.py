import rospy
import bitstring

import std_msgs.msg
from tauv_msgs.msg import DvlData as DvlDataMsg

class Ensemble:

    ID_SIZE = 2
    HEADER_SIZE = 6

    CHECKSUM_SIZE = 2
    CHECKSUM_MODULO = 65536

    HEADER_ID = '7f7f'
    FIXED_LEADER_ID = '0000'
    VARIABLE_LEADER_ID = '0080'
    BOTTOM_TRACK_DATA_ID = '0600'
    HR_BOTTOM_TRACK_DATA_ID = '5803'
    NAV_PARAMS_DATA_ID = '2013'

    def __init__(self, receive_time: rospy.Time):
        self.receive_time = receive_time
        self.depth = 0.0
        self.temperature = 0.0
        self.pressure = 0.0
        self.pressure_variance = 0.0
        self.depth_beam_1 = 0.0
        self.depth_beam_2 = 0.0
        self.depth_beam_3 = 0.0
        self.depth_beam_4 = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.velocity_error = 0.0

    def parse_header(self, d: bitstring.BitStream) -> int:
        self.header_data = d.bytes

        header_id = d.read('hex:16')
        if header_id != Ensemble.HEADER_ID:
            raise ValueError('Unexpected Header ID: {}'.format(header_id))

        self.ensemble_size = d.read('uintle:16')

        d.bytepos += 1

        self.datatype_count = d.read('uint:8')

        self.packet_size = self.ensemble_size - Ensemble.HEADER_SIZE + Ensemble.CHECKSUM_SIZE

        return self.packet_size

    def parse_packet(self, d: bitstring.BitStream):
        self.packet_data = d.bytes

        self._parse_datatype_offsets(d)

        for datatype_offset in self.datatype_offsets:
            self._parse_datatype(d, datatype_offset)

        self._validate_checksum(d)

    def is_valid(self) -> bool:
        return True

        if (self.depth is None
            or self.temperature is None
            or self.pressure is None
            or self.pressure_variance is None
            or self.depth_beam_1 is None
            or self.depth_beam_2 is None
            or self.depth_beam_3 is None
            or self.depth_beam_4 is None
            or self.velocity_x is None
            or self.velocity_y is None
            or self.velocity_z is None
            or self.velocity_error is None):
           return False

       # TODO: Add conditions to catch magic bad values

    def to_msg(self) -> DvlDataMsg:
        msg = DvlDataMsg()
        msg.header = std_msgs.msg.Header()

        msg.header.stamp = rospy.get_rostime()

        msg.depth = float(self.ensemble_number) 

        msg.temperature = self.temperature / 1e2

        msg.pressure = self.pressure * 1e1
        msg.pressure_variance = self.pressure_variance * 1e1

        msg.depth_beam_1 = self.depth_beam_1 / 1e2
        msg.depth_beam_2 = self.depth_beam_2 / 1e2
        msg.depth_beam_3 = self.depth_beam_3 / 1e2
        msg.depth_beam_4 = self.depth_beam_4 / 1e2

        msg.velocity_x = self.velocity_x / 1e3
        msg.velocity_y = self.velocity_y / 1e3
        msg.velocity_z = self.velocity_z / 1e3
        msg.velocity_error = self.velocity_error / 1e3

        return msg

    def _parse_datatype_offsets(self, d: bitstring.BitStream):
        d.bytepos = 0
        self.datatype_offsets = []

        for i in range(self.datatype_count):
            datatype_offset = d.read('uintle:16') - Ensemble.HEADER_SIZE - Ensemble.ID_SIZE + 1
            self.datatype_offsets.append(datatype_offset)

    def _parse_datatype(self, d: bitstring.BitStream, datatype_offset: int):
        d.bytepos = datatype_offset
        datatype_id = d.read('hex:16')

        if datatype_id == Ensemble.FIXED_LEADER_ID:
            self._parse_fixed_leader(d)
        elif datatype_id == Ensemble.VARIABLE_LEADER_ID:
            self._parse_variable_leader(d)
        elif datatype_id == Ensemble.BOTTOM_TRACK_DATA_ID:
            self._parse_bottom_track_data(d)
        elif datatype_id == Ensemble.HR_BOTTOM_TRACK_DATA_ID:
            self._parse_hr_bottom_track_data(d)
        elif datatype_id == Ensemble.NAV_PARAMS_DATA_ID:
            self._parse_nav_params_data(d)
        else:
            pass
            # raise ValueError('Unexpected Datatype ID: {}'.format(datatype_id))

    def _parse_fixed_leader(self, d: bitstring.BitStream):
        pass

    def _parse_variable_leader(self, d: bitstring.BitStream):
        self.ensemble_number = d.read('uintle:16')

        d.bytepos += 22

        self.temperature = d.read('intle:16')

        d.bytepos += 20

        self.pressure = d.read('uintle:32')

        self.pressure_variance = d.read('uintle:32')

    def _parse_bottom_track_data(self, d: bitstring.BitStream):
        d.bytepos += 14

        self.depth_beam_1 = d.read('uintle:16') # TODO: do we need to use the MSB?
        self.depth_beam_2 = d.read('uintle:16')
        self.depth_beam_3 = d.read('uintle:16')
        self.depth_beam_4 = d.read('uintle:16')

        self.velocity_x = d.read('intle:16')
        self.velocity_y = d.read('intle:16')
        self.velocity_z = d.read('intle:16')
        self.velocity_error = d.read('intle:16')

    def _parse_hr_bottom_track_data(self, d: bitstring.BitStream):
        pass

    def _parse_nav_params_data(self, d: bitstring.BitStream):
        pass

    def _validate_checksum(self, d: bitstring.BitStream):
        d.bytepos = self.packet_size - Ensemble.CHECKSUM_SIZE 

        self.expected_checksum = d.read('uintle:16')

        data_to_sum = self.header_data + self.packet_data[:-Ensemble.CHECKSUM_SIZE]

        self.checksum = sum(data_to_sum) % Ensemble.CHECKSUM_MODULO

        if self.checksum != self.expected_checksum:
            raise ValueError('Unexpected Checksum')
