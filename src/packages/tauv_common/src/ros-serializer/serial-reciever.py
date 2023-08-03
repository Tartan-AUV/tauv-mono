import serial
import serial.tools.list_ports

class Receiver:
    def __init__(self):
        self.port = self.find_port()
        if self.port == None:
            raise Exception("no serial port detected")

        self.serial = None
        self.payload = None
        self.checksum = None
        self.datalen = 10

    def find_port(self):
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                for p in str(port).split(' - '):
                    if '/dev/tty' in p and ('USB' in p or 'ACM' in p):
                        return p
            
            return None

    def open_serial_port(self):
        try:
            self.serial = serial.Serial(self.port, baudrate=9600, timeout=1)
        except serial.SerialException as e:
            raise Exception(f"Failed to open serial port {self.port}: {e}")

    def read_data(self):
        data = self.serial.read(self.datalen)

        while True:
            if len(data) == self.datalen and data[-1] == 10:
                self.payload = data[1:-3]
                self.checksum = data[-3]
                break
            else:
                self.serial.read(1)
                self.payload = None
                self.checksum = None

    def compute_checksum(self, data):
        total_bits_sum = sum(bin(byte).count('1') for byte in data)

        return total_bits_sum


    def spin(self):
        while True:
            self.read_data()

            if self.payload is not None and self.checksum is not None:
                calculated_checksum = self.compute_checksum(self.payload)
                if calculated_checksum == self.checksum:
                    print(f"Valid payload received: {self.payload}")
                else:
                    print(calculated_checksum, ": ", self.checksum, ": ", self.payload)
                    print("Checksum mismatch. Discarding data.")

if __name__ == "__main__":
    receiver = Receiver()
    receiver.open_serial_port()
    receiver.spin()
