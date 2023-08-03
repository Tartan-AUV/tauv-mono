import serial
import serial.tools.list_ports
import binascii
import rospy
from tauv_common.srv import AcousticSerialization, AcousticSerializationResponse

class Serializer():
    def __init__(self):
        self._baud = 9600
        self._port = self.find_port()
        if self._port is None:
            raise Exception("No serial port detected")

        self._ser = serial.Serial(self._port, self._baud)

    def find_port(self):
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            for p in str(port).split(' - '):
                if '/dev/tty' in p and ('USB' in p or 'ACM' in p):
                    return p

        return None

    def compute_checksum(self, data):
        total_bits_sum = sum(bin(byte).count('1') for byte in data)

        return total_bits_sum
        
    def serialize(self, msg):
        if type(msg) is not int:
            raise Exception("Unexpected type")
        
        chksum = self.compute_checksum(int.to_bytes(msg, 6, 'big'))

        raw_val = chksum + (msg << 8)

        return int.to_bytes(raw_val, 8, 'big')

    def send(self, msg):
        if msg is not None:
            self._ser.write(self.serialize(msg))

    def handle_send_message(self, request):
        if request.data:
            self.send(request.data)
        
            response = AcousticSerializationResponse()
            response.success = True
            response.message = str(self.serialize(request.data).hex())
        return response

if __name__ == "__main__":
    rospy.init_node("serializer_node")
    serializer = Serializer()
    service_name = "send_serial_message"
    rospy.Service(service_name, AcousticSerialization, serializer.handle_send_message)
    rospy.spin()
