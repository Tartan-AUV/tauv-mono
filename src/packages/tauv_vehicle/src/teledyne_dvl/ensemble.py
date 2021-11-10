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
    BOTTOM_TRACK_HR_VELOCITY_ID = '5803'

    def __init__(self):
        self.depth = None
        self.temperature = None
        self.pressure = None
        self.pressure_variance = None
        self.depth_beam_1 = None
        self.depth_beam_2 = None
        self.depth_beam_3 = None
        self.depth_beam_4 = None
        self.velocity_x = None
        self.velocity_y = None
        self.velocity_z = None
        self.velocity_error = None
        self.hr_velocity_x = None
        self.hr_velocity_y = None
        self.hr_velocity_z = None
        self.hr_velocity_error = None
        pass

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

    def to_msg(self) -> DvlDataMsg:
        msg = DvlDataMsg()
        msg.header = std_msgs.msg.Header()

        msg.header.stamp = rospy.get_rostime()

        msg.depth = self.depth or 0.0

        msg.temperature = self.temperature or 0.0

        msg.pressure = self.pressure or 0.0

        msg.pressure_variance = self.pressure_variance or 0.0

        msg.depth_beam_1 = self.depth_beam_1 or 0.0
        msg.depth_beam_2 = self.depth_beam_2 or 0.0
        msg.depth_beam_3 = self.depth_beam_3 or 0.0
        msg.depth_beam_4 = self.depth_beam_4 or 0.0

        msg.velocity_x = self.velocity_x or 0.0
        msg.velocity_y = self.velocity_y or 0.0
        msg.velocity_z = self.velocity_z or 0.0
        msg.velocity_error = self.velocity_error or 0.0

        msg.hr_velocity_x = self.hr_velocity_x or 0.0
        msg.hr_velocity_y = self.hr_velocity_y or 0.0
        msg.hr_velocity_z = self.hr_velocity_z or 0.0
        msg.hr_velocity_error = self.hr_velocity_error or 0.0

        return msg

    def _parse_datatype_offsets(self, d: bitstring.BitStream):
        d.bytepos = 0
        self.datatype_offsets = []

        for i in range(self.datatype_count):
            datatype_offset = d.read('uintle:16') - 7
            self.datatype_offsets.append(datatype_offset)

        print(self.datatype_offsets)

    def _parse_datatype(self, d: bitstring.BitStream, datatype_offset: int):
        d.bytepos = datatype_offset
        datatype_id = d.read('hex:16')

        print(datatype_id)

        if (datatype_id == Ensemble.FIXED_LEADER_ID):
            self._parse_fixed_leader(d)
        elif (datatype_id == Ensemble.VARIABLE_LEADER_ID):
            self._parse_variable_leader(d)
        elif (datatype_id == Ensemble.BOTTOM_TRACK_DATA_ID):
            self._parse_bottom_track_data(d)
        elif (datatype_id == Ensemble.BOTTOM_TRACK_HR_VELOCITY_ID):
            self._parse_bottom_track_hr_velocity(d)
        else:
            pass

    def _parse_fixed_leader(self, d: bitstring.BitStream):
        pass

    def _parse_variable_leader(self, d: bitstring.BitStream):
        self.ensemble_number = d.read('uintle:16')

        d.bytepos += 12

        self.depth = d.read('uintle:16') / 1e1

        d.bytepos += 8

        self.temperature = d.read('intle:16') / 1e2

        d.bytepos += 20

        self.pressure = d.read('uintle:32') * 1e1

        self.pressure_variance = d.read('uintle:32') * 1e1

    def _parse_bottom_track_data(self, d: bitstring.BitStream):
        d.bytepos += 14

        self.depth_beam_1 = d.read('uintle:16') / 1e2 # TODO: do we need to use the MSB?
        self.depth_beam_2 = d.read('uintle:16') / 1e2
        self.depth_beam_3 = d.read('uintle:16') / 1e2
        self.depth_beam_4 = d.read('uintle:16') / 1e2

        self.velocity_x = -d.read('intle:16') / 1e3
        self.velocity_y = -d.read('intle:16') / 1e3
        self.velocity_z = -d.read('intle:16') / 1e3
        self.velocity_error = -d.read('intle:16') / 1e3

    def _parse_bottom_track_hr_velocity(self, d: bitstring.BitStream):
        self.hr_velocity_x = d.read('intle:32') / 1e5
        self.hr_velocity_y = d.read('intle:32') / 1e5
        self.hr_velocity_z = d.read('intle:32') / 1e5
        self.hr_velocity_error = d.read('intle:32') / 1e5

    def _validate_checksum(self, d: bitstring.BitStream):
        d.bytepos = self.packet_size - 2 

        self.expected_checksum = d.read('uintle:16')

        data_to_sum = self.header_data + self.packet_data[:-2]

        self.checksum = sum(data_to_sum) % Ensemble.CHECKSUM_MODULO

        if self.checksum != self.expected_checksum:
            raise ValueError('Unexpected checksum')
