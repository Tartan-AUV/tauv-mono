import rospy
from tauv_msgs.srv import GetVoltage, GetVoltageRequest, GetVoltageResponse

from enum import Enum
import serial
import struct

class Commands(Enum):
    GetVoltage = 34

class Power:
    def __init__(self):
        self._srv: rospy.Service = rospy.Service("vehicle/power/get_voltage", GetVoltage, self._handle_get_voltage)

    def _handle_get_voltage(self, req : GetVoltageRequest):
        try: 
            ser = serial.Serial(rospy.get_param('~port'), baudrate=rospy.get_param('~baudrate'), timeout=.1)

            bytes = struct.pack('f', 34) #34 = 0x22

            bytes += b'\x00\x00\x00\x00\x00\x00\x00\x00'
            checksum = self._computeCheckSum(bytes)

            checksum_bytes = struct.pack('<I', checksum)
            bytes = bytes + checksum_bytes

            print(bytes)

            ser.write(bytes)

            startTime = rospy.Time.now()

            response = b''
            parsed = None

            while rospy.Time.now() - startTime < rospy.Duration.from_sec(1):
                response += ser.read(1)
                if len(response) == 11:
                    parsed = self._parse('f', response)
                    if parsed:
                        break
                    response = response[1:]
        
            return parsed

        except serial.SerialException as e:
            print(f"Error sending message: {e}")

    def _parse(self, format_string, response):

            checksum = self._computeCheckSum(response[:-2])

            if checksum != struct.unpack('H', response[9:11]): 
                return None
            
            commandID = struct.unpack('B', response[0])

            if commandID == 34:
                response = GetVoltageResponse()
                voltage = struct.unpack(format_string, response[1:5])
                response.voltage = voltage

                return response

    def _computeCheckSum(self, data_bytes, base=256, modulus=65521):
        hash_value = 0
        for byte in data_bytes:
            hash_value = (hash_value * base + byte) % modulus
        return hash_value 

    def start(self):
        rospy.spin()

def main():
    rospy.init_node('power')
    n = Power()
    n.start()
