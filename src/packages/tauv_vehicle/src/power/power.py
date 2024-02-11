import rospy
from tauv_msgs.srv import GetVoltage, GetVoltageRequest, GetVoltageResponse

from enum import Enum
import serial
import struct
import time

class Commands(Enum):
    GetVoltage = 34

class Power:
    def __init__(self):
        self._srv: rospy.Service = rospy.Service("vehicle/power/get_voltage", GetVoltage, self._handle_get_voltage)

    def _handle_get_voltage(self, req : GetVoltageRequest):
        try: 
            ser = serial.Serial(rospy.get_param('~port'), baudrate=rospy.get_param('~baudrate'), timeout=.1)

            bytes = struct.pack('c', b'\x22') #34 = 0x22

            bytes += b'\x00\x00\x00\x00\x00\x00\x00\x00'
            checksum = self._computeCheckSum(bytes)

            checksum_bytes = struct.pack('>H', checksum)
            bytes = bytes + checksum_bytes

            print(bytes.hex())  
            time.sleep(2)
            ser.write(bytes)
            
            startTime = rospy.Time.now()
            
            response = b''
            parsed = None
            
            while rospy.Time.now() - startTime < rospy.Duration.from_sec(1):
                byteRead = ser.read(1)
                response += byteRead
                print("byteRead: "+ str(byteRead)+" and response: "+str(response))
                if len(response) == 11:
                    print("got here")
                    parsed = self._parse('>f', response)
                    if parsed:
                        break
                    response = response[1:]
        
            return parsed

        except serial.SerialException as e:
            print(f"Error sending message: {e}")

    def _parse(self, format_string, response):
            checksum = self._computeCheckSum(response[:-2])
            checkSumRecieved = struct.unpack('H', response[9:11])
            if checksum != checkSumRecieved[0]: 
                print("recieved checksum: "+str(checkSumRecieved))
                return None
            
            commandID = struct.unpack('B', response[0:1])[0]

            if commandID == 34:
                voltage = struct.unpack(format_string, response[1:5])[0]
                print("voltage: "+str(voltage))
                print("voltage bytes: "+str(response[1:5].hex()))
                return GetVoltageResponse(voltage)

    def _computeCheckSum(self, data_bytes, base=256, modulus=65521):
        hash_value = 0
        for byte in data_bytes:
            hash_value = (hash_value * base + byte) % modulus

        #print(hash_value)
        return hash_value 

    def start(self):
        rospy.spin()

def main():
    rospy.init_node('power')
    n = Power()
    n.start()
