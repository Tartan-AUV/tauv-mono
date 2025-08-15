import re

class PCA9685Exception(Exception):
    pass

class Registers(object):
    MODE_1 = 0x00
    MODE_2 = 0x01
    LED_STRIP_START = 0x06
    PRE_SCALE = 0xFE

class Mode1(object):
    RESTART = 7
    EXTCLK = 6
    AI = 5
    SLEEP = 4
    SUB1 = 3
    SUB2 = 2
    SUB3 = 1
    ALLCALL = 0

class Mode2(object):
    INVRT = 4
    OCH = 3
    OUTDRV = 2
    OUTNE_1 = 1
    OUTNE_0 = 0

def value_low(val):
    return val & 0xFF

def value_high(val):
    return (val >> 8) & 0xFF

class PCA9685(object):

    ranges = dict(
        pwm_frequency = (24, 1526),
        led_number = (0, 15),
        led_value = (0, 4095),
        register_value = (0, 255),
    )

    def __init__(self, address, bus=None):
        self.__address = address
        self.__bus = bus
        self.__oscillator_clock = 25000000

    @property
    def mode_1(self):
        return self.read(Registers.MODE_1)

    @property
    def bus(self):
        return self.__bus

    def get_led_register_from_name(self, name):
        res = re.match('^led_([0-9]{1,2})$', name)
        if res is None:
            raise AttributeError("Unknown attribute: '%s'" % name)
        led_num = int(res.group(1))
        if led_num < 0 or led_num > 15:
            raise AttributeError("Unknown attribute: '%s'" % name)
        return self.calc_led_register(led_num)

    def calc_led_register(self, led_num):
        start = Registers.LED_STRIP_START + 2
        return start + (led_num * 4)

    def __check_range(self, type, value):
        range = self.ranges[type]
        if value < range[0]:
            raise PCA9685Exception("%s must be greater than %s, got %s" % (type, range[0], value))
        if value > range[1]:
            raise PCA9685Exception("%s must be less than %s, got %s" % (type, range[1], value))

    def set_pwm(self, led_num, value):
        self.__check_range('led_number', led_num)
        self.__check_range('led_value', value)

        register_low = self.calc_led_register(led_num)

        self.write(register_low, value_low(value))
        self.write(register_low + 1, value_high(value))

    def __get_led_value(self, register_low):
        low = self.read(register_low)
        high = self.read(register_low + 1)
        return low + (high * 256)

    def get_pwm(self, led_num):
        self.__check_range('led_number', led_num)
        register_low = self.calc_led_register(led_num)
        return self.__get_led_value(register_low)

    def __getattr__(self, name):
        register_low = self.get_led_register_from_name(name)
        return self.__get_led_value(register_low)

    def sleep(self):
        self.write(Registers.MODE_1, self.mode_1 | (1 << Mode1.SLEEP))

    def wake(self):
        self.write(Registers.MODE_1, self.mode_1 & (255 - (1 << Mode1.SLEEP)))

    def write(self, reg, value):
        self.__check_range('register_value', value)
        self.__bus.write_byte_data(self.__address, reg, value)

    def read(self, reg):
        return self.__bus.read_byte_data(self.__address, reg)

    def calc_pre_scale(self, frequency):
        return int(round(self.__oscillator_clock / (4096.0 * frequency)) - 1)

    def set_pwm_frequency(self, value):
        self.__check_range('pwm_frequency', value)
        reg_val = self.calc_pre_scale(value)
        self.sleep()
        self.write(Registers.PRE_SCALE, reg_val)
        self.wake()

    def calc_frequency(self, prescale):
        return int(round(self.__oscillator_clock / ((prescale + 1) * 4096.0)))

    def get_pwm_frequency(self):
        return self.calc_frequency(self.read(Registers.PRE_SCALE))
