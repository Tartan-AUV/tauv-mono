try:
    import smbus2 as smbus
except:
    import smbus
    
from time import sleep

# Models
MODEL_02BA = 0
MODEL_30BA = 1
MODEL_UNKNOWN = 255

MS5837_02BA_MAX_SENSITIVITY = 49000
MS5837_02BA_30BA_SEPARATION = 37000
MS5837_30BA_MIN_SENSITIVITY = 26000

# Oversampling options
OSR_256  = 0
OSR_512  = 1
OSR_1024 = 2
OSR_2048 = 3
OSR_4096 = 4
OSR_8192 = 5

# kg/m^3 convenience
DENSITY_FRESHWATER = 997
DENSITY_SALTWATER = 1029

# Conversion factors (from native unit, mbar)
UNITS_Pa     = 100.0
UNITS_hPa    = 1.0
UNITS_kPa    = 0.1
UNITS_mbar   = 1.0
UNITS_bar    = 0.001
UNITS_atm    = 0.000986923
UNITS_Torr   = 0.750062
UNITS_psi    = 0.014503773773022

# Valid units
UNITS_Centigrade = 1
UNITS_Farenheit  = 2
UNITS_Kelvin     = 3

    
class MS5837(object):
    
    # Registers
    _MS5837_ADDR             = 0x76  
    _MS5837_RESET            = 0x1E
    _MS5837_ADC_READ         = 0x00
    _MS5837_PROM_READ        = 0xA0
    _MS5837_CONVERT_D1_256   = 0x40
    _MS5837_CONVERT_D2_256   = 0x50
    
    def __init__(self, model=MODEL_UNKNOWN, bus=None):
        self._model = model
        self._bus = bus
        self._fluidDensity = DENSITY_FRESHWATER
        self._pressure = 0
        self._temperature = 0
        self._D1 = 0
        self._D2 = 0
        
    def init(self):
        if self._bus is None:
            return False
        
        self._bus.write_byte(self._MS5837_ADDR, self._MS5837_RESET)
        
        # Wait for reset to complete
        sleep(0.01)
        
        self._C = []
        
        # Read calibration values and CRC
        for i in range(7):
            c = self._bus.read_word_data(self._MS5837_ADDR, self._MS5837_PROM_READ + 2*i)
            c =  ((c & 0xFF) << 8) | (c >> 8)
            self._C.append(c)
                        
        crc = (self._C[0] & 0xF000) >> 12
        if crc != self._crc4(self._C):
            return False

        if self._model == MODEL_UNKNOWN:
            self.auto_detect_model()
        
        return True

    def auto_detect_model(self):
        pressure_sensitivity = self._C[1]
        if MS5837_30BA_MIN_SENSITIVITY <= pressure_sensitivity <= MS5837_02BA_30BA_SEPARATION:
            self._model = MODEL_30BA
        elif MS5837_02BA_30BA_SEPARATION < pressure_sensitivity <= MS5837_02BA_MAX_SENSITIVITY:
            self._model = MODEL_02BA
        else:
            self._model = MODEL_UNKNOWN
        
    def read(self, oversampling=OSR_8192):
        if self._bus is None:
            return False
        
        if oversampling < OSR_256 or oversampling > OSR_8192:
            return False
        
        # Request D1 conversion (pressure)
        self._bus.write_byte(self._MS5837_ADDR, self._MS5837_CONVERT_D1_256 + 2*oversampling)
    
        # Maximum conversion time increases linearly with oversampling
        sleep(2.5e-6 * 2**(8+oversampling))
        
        d = self._bus.read_i2c_block_data(self._MS5837_ADDR, self._MS5837_ADC_READ, 3)
        self._D1 = d[0] << 16 | d[1] << 8 | d[2]
        
        # Request D2 conversion (temperature)
        self._bus.write_byte(self._MS5837_ADDR, self._MS5837_CONVERT_D2_256 + 2*oversampling)
    
        # As above
        sleep(2.5e-6 * 2**(8+oversampling))
 
        d = self._bus.read_i2c_block_data(self._MS5837_ADDR, self._MS5837_ADC_READ, 3)
        self._D2 = d[0] << 16 | d[1] << 8 | d[2]

        # Calculate compensated pressure and temperature
        self._calculate()
        
        return True
    
    def setFluidDensity(self, denisty):
        self._fluidDensity = denisty
        
    def pressure(self, conversion=UNITS_mbar):
        return self._pressure * conversion
        
    def temperature(self, conversion=UNITS_Centigrade):
        degC = self._temperature / 100.0
        if conversion == UNITS_Farenheit:
            return (9.0/5.0)*degC + 32
        elif conversion == UNITS_Kelvin:
            return degC + 273
        return degC
        
    def depth(self):
        return (self.pressure(UNITS_Pa)-101300)/(self._fluidDensity*9.80665)
    
    def altitude(self):
        return (1-pow((self.pressure()/1013.25),.190284))*145366.45*.3048        
    
    def _calculate(self):
        OFFi = 0
        SENSi = 0
        Ti = 0

        dT = self._D2-self._C[5]*256
        if self._model == MODEL_02BA:
            SENS = self._C[1]*65536+(self._C[3]*dT)/128
            OFF = self._C[2]*131072+(self._C[4]*dT)/64
            self._pressure = (self._D1*SENS/(2097152)-OFF)/(32768)
        elif self._model == MODEL_30BA:
            SENS = self._C[1]*32768+(self._C[3]*dT)/256
            OFF = self._C[2]*65536+(self._C[4]*dT)/128
            self._pressure = (self._D1*SENS/(2097152)-OFF)/(8192)
        else:
            raise NotImplementedError("Cannot calculate pressure for unknown model type!")
        
        self._temperature = 2000+dT*self._C[6]/8388608

        # Second order compensation
        if self._model == MODEL_02BA:
            if (self._temperature/100) < 20:
                Ti = (11*dT*dT)/(34359738368)
                OFFi = (31*(self._temperature-2000)*(self._temperature-2000))/8
                SENSi = (63*(self._temperature-2000)*(self._temperature-2000))/32
                
        else:
            if (self._temperature/100) < 20:
                Ti = (3*dT*dT)/(8589934592)
                OFFi = (3*(self._temperature-2000)*(self._temperature-2000))/2
                SENSi = (5*(self._temperature-2000)*(self._temperature-2000))/8
                if (self._temperature/100) < -15:
                    OFFi = OFFi+7*(self._temperature+1500)*(self._temperature+1500)
                    SENSi = SENSi+4*(self._temperature+1500)*(self._temperature+1500)
            elif (self._temperature/100) >= 20:
                Ti = 2*(dT*dT)/(137438953472)
                OFFi = (1*(self._temperature-2000)*(self._temperature-2000))/16
                SENSi = 0
        
        OFF2 = OFF-OFFi
        SENS2 = SENS-SENSi
        
        if self._model == MODEL_02BA:
            self._temperature = (self._temperature-Ti)
            self._pressure = (((self._D1*SENS2)/2097152-OFF2)/32768)/100.0
        else:
            self._temperature = (self._temperature-Ti)
            self._pressure = (((self._D1*SENS2)/2097152-OFF2)/8192)/10.0   
        
    def _crc4(self, n_prom):
        n_rem = 0
        
        n_prom[0] = ((n_prom[0]) & 0x0FFF)
        n_prom.append(0)
    
        for i in range(16):
            if i%2 == 1:
                n_rem ^= ((n_prom[i>>1]) & 0x00FF)
            else:
                n_rem ^= (n_prom[i>>1] >> 8)
                
            for n_bit in range(8,0,-1):
                if n_rem & 0x8000:
                    n_rem = (n_rem << 1) ^ 0x3000
                else:
                    n_rem = (n_rem << 1)

        n_rem = ((n_rem >> 12) & 0x000F)
        
        self.n_prom = n_prom
        self.n_rem = n_rem
    
        return n_rem ^ 0x00
    
class MS5837_30BA(MS5837):
    def __init__(self, bus=None):
        MS5837.__init__(self, MODEL_30BA, bus)
        
class MS5837_02BA(MS5837):
    def __init__(self, bus=None):
        MS5837.__init__(self, MODEL_02BA, bus)
