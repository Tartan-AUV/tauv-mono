#include "MS5837.h"
#include <cmath>

namespace TAUV {

const uint8_t MS5837_ADDR = 0x76;
const uint8_t MS5837_RESET = 0x1E;
const uint8_t MS5837_ADC_READ = 0x00;
const uint8_t MS5837_PROM_READ = 0xA0;
const uint8_t MS5837_CONVERT_D1_8192 = 0x4A;
const uint8_t MS5837_CONVERT_D2_8192 = 0x5A;

const float MS5837::Pa = 100.0f;
const float MS5837::bar = 0.001f;
const float MS5837::mbar = 1.0f;

const uint8_t MS5837::MS5837_30BA = 0;
const uint8_t MS5837::MS5837_02BA = 1;
const uint8_t MS5837::MS5837_UNRECOGNISED = 255;

const uint8_t MS5837_02BA01 = 0x00; // Sensor version: From MS5837_02BA datasheet Version PROM Word 0
const uint8_t MS5837_02BA21 = 0x15; // Sensor version: From MS5837_02BA datasheet Version PROM Word 0
const uint8_t MS5837_30BA26 = 0x1A; // Sensor version: From MS5837_30BA datasheet Version PROM Word 0


MS5837::MS5837() {
    fluidDensity = 1029;
    _hi2c = nullptr;
    conversion_timer = nullptr;
}

MS5837::~MS5837() {
    // Delete timer if it exists
    if (conversion_timer != nullptr) {
        xTimerDelete(conversion_timer, portMAX_DELAY);
        conversion_timer = nullptr;
    }
}

bool MS5837::begin(I2C_HandleTypeDef *hi2c) {
    return (init(hi2c));
}

bool MS5837::init(I2C_HandleTypeDef *hi2c) {
    _hi2c = hi2c; // Store I2C handle

    if (_hi2c == nullptr) {
        return false;
    }

    // Reset the MS5837, per datasheet
    uint8_t resetCmd = MS5837_RESET;
    if (HAL_I2C_Master_Transmit(_hi2c, MS5837_ADDR << 1, &resetCmd, 1, HAL_MAX_DELAY) != HAL_OK) {
        return false;
    }

    // Wait for reset to complete
    HAL_Delay(10);

    // Read calibration values and CRC
    for (uint8_t i = 0; i < 7; i++) {
        uint8_t cmd = MS5837_PROM_READ + i * 2;
        if (HAL_I2C_Master_Transmit(_hi2c, MS5837_ADDR << 1, &cmd, 1, HAL_MAX_DELAY) != HAL_OK) {
            return false;
        }

        uint8_t data[2];
        if (HAL_I2C_Master_Receive(_hi2c, MS5837_ADDR << 1, data, 2, HAL_MAX_DELAY) != HAL_OK) {
            return false;
        }

        C[i] = (data[0] << 8) | data[1];
    }

    // Verify that data is correct with CRC
    uint8_t crcRead = C[0] >> 12;
    uint8_t crcCalculated = crc4(C);

    if (crcCalculated != crcRead) {
        return false; // CRC fail
    }

    uint8_t version = (C[0] >> 5) & 0x7F; // Extract the sensor version from PROM Word 0

    // Set _model according to the sensor version
    if (version == MS5837_02BA01) {
        _model = MS5837_02BA;
    } else if (version == MS5837_02BA21) {
        _model = MS5837_02BA;
    } else if (version == MS5837_30BA26) {
        _model = MS5837_30BA;
    } else {
        _model = MS5837_UNRECOGNISED;
    }
    
    // Create a FreeRTOS timer for handling conversion timing
    if (conversion_timer == nullptr) {
        conversion_timer = xTimerCreate(
            "MS5837_Timer",                  // Timer name
            pdMS_TO_TICKS(CONVERSION_TIME_MS), // Default period (will be updated when starting conversion)
            pdFALSE,                         // Auto-reload (one-shot)
            this,                            // Timer ID is a pointer to this instance
            conversionTimerCallback          // Callback function
        );
        
        if (conversion_timer == nullptr) {
            return false; // Failed to create timer
        }
    }
    
    // Initialize state
    conversion_state = ConversionState::IDLE;
    data_ready = false;
    
    return true;
}

void MS5837::setModel(uint8_t model) {
    _model = model;
}

uint8_t MS5837::getModel() {
    return (_model);
}

void MS5837::setFluidDensity(float density) {
    fluidDensity = density;
}

bool MS5837::requestConversion(MS5837::Oversampling osr) {
    if (_hi2c == nullptr || conversion_timer == nullptr) {
        return false;
    }
    
    // Store the current oversampling rate
    current_oversampling = osr;
    
    // Start the conversion sequence with temperature
    return requestTemperature();
}

bool MS5837::requestTemperature() {
    if (_hi2c == nullptr) {
        return false;
    }
    
    // Request temperature conversion
    uint8_t cmd = oversampling_command_map_temperature[static_cast<size_t>(current_oversampling)];
    if (HAL_I2C_Master_Transmit(_hi2c, MS5837_ADDR << 1, &cmd, 1, HAL_MAX_DELAY) != HAL_OK) {
        conversion_state = ConversionState::IDLE;
        return false;
    }
    
    // Update state
    conversion_state = ConversionState::TEMP_REQUESTED;
    
    // Start timer for conversion completion
    xTimerChangePeriod(conversion_timer, pdMS_TO_TICKS(CONVERSION_TIME_MS), portMAX_DELAY);
    xTimerStart(conversion_timer, portMAX_DELAY);
    
    return true;
}

bool MS5837::requestPressure() {
    if (_hi2c == nullptr) {
        return false;
    }
    
    // Request pressure conversion
    uint8_t cmd = oversampling_command_map_pressure[static_cast<size_t>(current_oversampling)];
    if (HAL_I2C_Master_Transmit(_hi2c, MS5837_ADDR << 1, &cmd, 1, HAL_MAX_DELAY) != HAL_OK) {
        conversion_state = ConversionState::IDLE;
        return false;
    }
    
    // Update state
    conversion_state = ConversionState::PRESSURE_REQUESTED;
    
    // Start timer for conversion completion
    xTimerChangePeriod(conversion_timer, pdMS_TO_TICKS(CONVERSION_TIME_MS), portMAX_DELAY);
    xTimerStart(conversion_timer, portMAX_DELAY);
    
    return true;
}

bool MS5837::readConversion() {
    if (_hi2c == nullptr) {
        return false;
    }
    
    // Read the conversion result
    uint8_t cmd = MS5837_ADC_READ;
    if (HAL_I2C_Master_Transmit(_hi2c, MS5837_ADDR << 1, &cmd, 1, HAL_MAX_DELAY) != HAL_OK) {
        conversion_state = ConversionState::IDLE;
        return false;
    }
    
    uint8_t data[3];
    if (HAL_I2C_Master_Receive(_hi2c, MS5837_ADDR << 1, data, 3, HAL_MAX_DELAY) != HAL_OK) {
        conversion_state = ConversionState::IDLE;
        return false;
    }
    
    uint32_t result = ((uint32_t)data[0] << 16) | ((uint32_t)data[1] << 8) | data[2];
    
    // Store the result based on current state
    if (conversion_state == ConversionState::TEMP_REQUESTED) {
        D2_temp = result;
        // After reading temperature, request pressure
        return requestPressure();
    } else if (conversion_state == ConversionState::PRESSURE_REQUESTED) {
        D1_pres = result;
        // Calculate values once we have both pressure and temperature
        calculate();
        // Mark data as ready
        data_ready = true;
        // Return to idle state
        conversion_state = ConversionState::IDLE;
        return true;
    }
    
    return false;
}

bool MS5837::read() {
    // Simple getter that returns the current data valid state
    // and resets the flag once read
    bool was_valid = data_ready;
    data_ready = false;
    return was_valid;
}

void MS5837::conversionTimerCallback(TimerHandle_t timer) {
    // Get the MS5837 instance from the timer ID
    MS5837* instance = static_cast<MS5837*>(pvTimerGetTimerID(timer));
    if (instance != nullptr) {
        // Read the result and continue the conversion sequence
        instance->readConversion();
    }
}

void MS5837::calculate() {
    // Given C1-C6 and D1, D2, calculated TEMP and P
    // Do conversion first and then second order temp compensation

    int32_t dT = 0;
    int64_t SENS = 0;
    int64_t OFF = 0;
    int32_t SENSi = 0;
    int32_t OFFi = 0;
    int32_t Ti = 0;
    int64_t OFF2 = 0;
    int64_t SENS2 = 0;

    // Terms called
    dT = D2_temp - uint32_t(C[5]) * 256l;
    if (_model == MS5837_02BA) {
        SENS = int64_t(C[1]) * 65536l + (int64_t(C[3]) * dT) / 128l;
        OFF = int64_t(C[2]) * 131072l + (int64_t(C[4]) * dT) / 64l;
        P = (D1_pres * SENS / (2097152l) - OFF) / (32768l);
    } else {
        SENS = int64_t(C[1]) * 32768l + (int64_t(C[3]) * dT) / 256l;
        OFF = int64_t(C[2]) * 65536l + (int64_t(C[4]) * dT) / 128l;
        P = (D1_pres * SENS / (2097152l) - OFF) / (8192l);
    }

    // Temp conversion
    TEMP = 2000l + int64_t(dT) * C[6] / 8388608LL;

    // Second order compensation
    if (_model == MS5837_02BA) {
        if ((TEMP / 100) < 20) {         // Low temp
            Ti = (11 * int64_t(dT) * int64_t(dT)) / (34359738368LL);
            OFFi = (31 * (TEMP - 2000) * (TEMP - 2000)) / 8;
            SENSi = (63 * (TEMP - 2000) * (TEMP - 2000)) / 32;
        }
    } else {
        if ((TEMP / 100) < 20) {         // Low temp
            Ti = (3 * int64_t(dT) * int64_t(dT)) / (8589934592LL);
            OFFi = (3 * (TEMP - 2000) * (TEMP - 2000)) / 2;
            SENSi = (5 * (TEMP - 2000) * (TEMP - 2000)) / 8;
            if ((TEMP / 100) < -15) {    // Very low temp
                OFFi = OFFi + 7 * (TEMP + 1500l) * (TEMP + 1500l);
                SENSi = SENSi + 4 * (TEMP + 1500l) * (TEMP + 1500l);
            }
        } else if ((TEMP / 100) >= 20) {    // High temp
            Ti = 2 * (dT * dT) / (137438953472LL);
            OFFi = (1 * (TEMP - 2000) * (TEMP - 2000)) / 16;
            SENSi = 0;
        }
    }

    OFF2 = OFF - OFFi;           // Calculate pressure and temp second order
    SENS2 = SENS - SENSi;

    TEMP = (TEMP - Ti);

    if (_model == MS5837_02BA) {
        P = (((D1_pres * SENS2) / 2097152l - OFF2) / 32768l);
    } else {
        P = (((D1_pres * SENS2) / 2097152l - OFF2) / 8192l);
    }
}

float MS5837::pressure(float conversion) {
    if (_model == MS5837_02BA) {
        return P * conversion / 100.0f;
    } else {
        return P * conversion / 10.0f;
    }
}

float MS5837::temperature() {
    return TEMP / 100.0f;
}

float MS5837::depth() {
    return (pressure(MS5837::Pa) - 101300) / (fluidDensity * 9.80665);
}

float MS5837::altitude() {
    return (1 - pow((pressure() / 1013.25), .190284)) * 145366.45 * .3048;
}

uint8_t MS5837::crc4(uint16_t n_prom[]) {
    uint16_t n_rem = 0;

    n_prom[0] = ((n_prom[0]) & 0x0FFF);
    n_prom[7] = 0;

    for (uint8_t i = 0; i < 16; i++) {
        if (i % 2 == 1) {
            n_rem ^= (uint16_t)((n_prom[i >> 1]) & 0x00FF);
        } else {
            n_rem ^= (uint16_t)(n_prom[i >> 1] >> 8);
        }
        for (uint8_t n_bit = 8; n_bit > 0; n_bit--) {
            if (n_rem & 0x8000) {
                n_rem = (n_rem << 1) ^ 0x3000;
            } else {
                n_rem = (n_rem << 1);
            }
        }
    }

    n_rem = ((n_rem >> 12) & 0x000F);

    return n_rem ^ 0x00;
}

} // namespace TAUV
