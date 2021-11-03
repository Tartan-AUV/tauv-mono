import rospy
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
        pass

    def parse_header(self, d):
        self.header_data = d.bytes

        header_id = d.read('hex:16')
        if header_id != Ensemble.HEADER_ID:
            raise ValueError('Unexpected Header ID: {}'.format(header_id))

        self.ensemble_size = d.read('uintle:16')
        
        d.bytepos += 1

        self.datatype_count = d.read('uint:8')

        self.packet_size = ensemble_size - Ensemble.HEADER_SIZE + Ensemble.CHECKSUM_SIZE

        return self.packet_size 

    def parse_packet(self, d):
        self.packet_data = d.bytes

        self._parse_datatype_offsets(d)

        for datatype_offset in self.datatype_offsets:
            self._parse_datatype(d, datatype_offset)

        self._validate_checksum(d)

    def to_msg(self):
        msg = DvlDataMsg()
        msg.header = Header()

        msg.header.stamp = rospy.get_rostime()

        msg.depth = self.depth

        msg.temperature = self.temperature

        msg.pressure = self.pressure

        msg.pressure_variance = self.pressure_variance

        msg.depth_beam_1 = self.depth_beam_1
        msg.depth_beam_2 = self.depth_beam_2
        msg.depth_beam_3 = self.depth_beam_3
        msg.depth_beam_4 = self.depth_beam_4

        msg.velocity_x = self.velocity_x
        msg.velocity_y = self.velocity_y
        msg.velocity_z = self.velocity_z
        msg.velocity_error = self.velocity_error

        msg.hr_velocity_x = self.hr_velocity_x
        msg.hr_velocity_y = self.hr_velocity_y
        msg.hr_velocity_z = self.hr_velocity_z
        msg.hr_velocity_error = self.hr_velocity_error

        return msg

    def _parse_datatype_offsets(self, d):
        d.bytepos = 0

        for i in range(self.datatype_count):
            datatype_offset = d.read('uintle:16') - ENSEMBLE_SIZE.HEADER_SIZE
            self.datatype_offsets.append(datatype_offset)

    def _parse_datatype(self, d, datatype_offset):
        d.bytepos = datatype_offset
        datatype_id = d.read('hex:32')

        if (datatype_id == Ensemble.FIXED_LEADER_ID):
            self._parse_fixed_leader(d)
        elif (datatype_id == Ensemble.VARIABLE_LEADER_ID):
            self._parse_variable_leader(d)
        elif (datatype_id == Ensemble.BOTTOM_TRACK_DATA_ID):
            self._parse_bottom_track_data(d)
        elif (datatype_id == Ensemble.BOTTOM_TRACK_HR_VELOCITY_ID):
            self._parse_bottom_track_hr_velocity(d)
        else:
            raise ValueError('Unexpected Datatype ID: {}'.format(datatype_id))

    def _parse_fixed_leader(self, d):
        pass

    def _parse_variable_leader(self, d):
        self.ensemble_number = d.read('uintle:16')

        d.bytepos += 12

        self.depth = d.read('uintle:16') / 1e1

        d.bytepos += 8

        self.temperature = d.read('intle:16') / 1e2

        d.bytepos += 20

        self.pressure = d.read('uintle:32') * 1e1

        self.pressure_variance = d.read('uintle:32') * 1e1

    def _parse_bottom_track_data(self, d):
        d.bytepos += 14

        self.depth_beam_1 = d.read('uintle:16') / 1e2 # TODO: do we need to use the MSB?
        self.depth_beam_2 = d.read('uintle:16') / 1e2
        self.depth_beam_3 = d.read('uintle:16') / 1e2
        self.depth_beam_4 = d.read('uintle:16') / 1e2

        self.velocity_x = -d.read('intle:16') / 1e3
        self.velocity_y = -d.read('intle:16') / 1e3
        self.velocity_z = -d.read('intle:16') / 1e3
        self.velocity_error = -d.read('intle:16') / 1e3

    def _parse_bottom_track_hr_velocity(self, d):
        self.hr_velocity_x = d.read('intle:32') / 1e5
        self.hr_velocity_y = d.read('intle:32') / 1e5
        self.hr_velocity_z = d.read('intle:32') / 1e5
        self.hr_velocity_error = d.read('intle:32') / 1e5

    def _validate_checksum(self, d):
        d.bytepos = self.packet_size - 2 

        self.expected_checksum = d.read('uintle:16')

        data_to_sum = self.header_data.append(self.packet_data[:-2])

        self.checksum = sum(map(ord, data_to_sum)) % Ensemble.CHECKSUM_MODULO

        if self.checksum != self.expected_checksum:
            raise ValueError('Unexpected checksum')
