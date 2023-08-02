#!/usr/bin/python3

import rospy
import serial

class Reciever():
    def __init__(self):
        self._chksum_msk = 0xFF
        self._data_msk = 0x00FFFFFFFFFFFF00
        self._data_max = 0xFFFFFFFFFFFF
        self._baud = 115200
    
    def bit_sum(self, data):
        checksum = 0

        while(data != 0):
            byte_data = bytes([data & 0xFF])

            print(bytes.hex(byte_data))

            for byte in byte_data:
                checksum += byte

            data = data >> 8

        return int(checksum / 6)
    
    def checksum(self, msg):
        sum = msg & self._chksum_msk
        data = (msg & self._data_msk) >> 8

        return sum == self.bit_sum(data)
    
    def recover(self, msg):
        return bytes.hex(msg)
    
    def spin(self):
        while True:
            msg = serial.Serial('/dev/arduino', self._baud, timeout=.05)

            if self.checksum(msg):
                data = self.recover(msg)
                print(data)

                # do something with data
            else:
                print("message checksum fail")

class Sender():
    def __init__(self):
        pass


def main():
    print(serial.Serial'/dev/)
    x = Reciever()
    print(x.checksum(0xFFFFF0FFFFFFFC))

if __name__ == "__main__":
    main()

