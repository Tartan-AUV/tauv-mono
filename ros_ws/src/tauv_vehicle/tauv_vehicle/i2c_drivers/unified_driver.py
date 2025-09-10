try:
    import smbus2 as smbus
except:
    import smbus

from .ms5837 import (
    MODEL_02BA,
    MODEL_30BA,
    MODEL_UNKNOWN,
    MS5837,
    OSR_8192,
    UNITS_Centigrade,
    UNITS_mbar,
)
from .pca9685 import PCA9685


class UnifiedI2CDriver:
    def __init__(self, bus_number=1, pca9685_address=0x40, ms5837_model=MODEL_UNKNOWN):
        """
        Initialize unified I2C driver for MS5837 and PCA9685

        Args:
            bus_number: I2C bus number (default 1)
            pca9685_address: I2C address for PCA9685 (default 0x40)
            ms5837_model: MS5837 model (MODEL_02BA, MODEL_30BA, or MODEL_UNKNOWN for auto-detect)
        """
        self._bus = smbus.SMBus(bus_number)

        self.ms5837 = MS5837(model=ms5837_model, bus=self._bus)
        self.pca9685 = PCA9685(address=pca9685_address, bus=self._bus)

        self._ms5837_initialized = False
        self._pca9685_initialized = False

    def init_ms5837(self):
        """Initialize MS5837 depth sensor"""
        result = self.ms5837.init()
        self._ms5837_initialized = result
        return result

    def init_pca9685(self):
        """Initialize PCA9685 PWM controller"""
        try:
            self.pca9685.wake()
            self._pca9685_initialized = True
            return True
        except:
            self._pca9685_initialized = False
            return False

    def init_all(self):
        """Initialize both devices"""
        ms5837_ok = self.init_ms5837()
        pca9685_ok = self.init_pca9685()
        return ms5837_ok and pca9685_ok

    # MS5837 methods
    def read_depth_sensor(self, oversampling=OSR_8192):
        """Read pressure and temperature from MS5837"""
        if not self._ms5837_initialized:
            return False
        return self.ms5837.read(oversampling)

    def get_pressure(self, conversion=UNITS_mbar):
        """Get pressure reading"""
        return self.ms5837.pressure(conversion)

    def get_temperature(self, conversion=UNITS_Centigrade):
        """Get temperature reading"""
        return self.ms5837.temperature(conversion)

    def get_depth(self):
        """Get depth reading"""
        return self.ms5837.depth()

    def get_altitude(self):
        """Get altitude reading"""
        return self.ms5837.altitude()

    def set_fluid_density(self, density):
        """Set fluid density for depth calculations"""
        self.ms5837.setFluidDensity(density)

    # PCA9685 methods
    def set_pwm(self, channel, value):
        """Set PWM value for a channel (0-15, value 0-4095)"""
        if not self._pca9685_initialized:
            return False
        self.pca9685.set_pwm(channel, value)
        return True

    def get_pwm(self, channel):
        """Get PWM value for a channel"""
        if not self._pca9685_initialized:
            return None
        return self.pca9685.get_pwm(channel)

    def set_pwm_frequency(self, frequency):
        """Set PWM frequency in Hz (24-1526)"""
        if not self._pca9685_initialized:
            return False
        self.pca9685.set_pwm_frequency(frequency)
        return True

    def get_pwm_frequency(self):
        """Get current PWM frequency"""
        if not self._pca9685_initialized:
            return None
        return self.pca9685.get_pwm_frequency()

    def close(self):
        """Close I2C bus connection"""
        if self._bus:
            self._bus.close()


class UnifiedI2CDriver_02BA(UnifiedI2CDriver):
    def __init__(self, bus_number=1, pca9685_address=0x40):
        super().__init__(bus_number, pca9685_address, MODEL_02BA)


class UnifiedI2CDriver_30BA(UnifiedI2CDriver):
    def __init__(self, bus_number=1, pca9685_address=0x40):
        super().__init__(bus_number, pca9685_address, MODEL_30BA)
