/* TAUV RTVC */
/* Depth Sensor Task */
/* Author: Shayaan Gandhi*/

#include "depthsensor.h"
#include "main.h"
#include "vehicle_config.h"
#include "vesc.h"


uint8_t calibration_coeff[2];


void depth_sensor_init()
{
    
    const uint8_t resetByte = 0x1E;
    const uint8_t PROM_Calibration = 0xA0;


    HAL_StatusTypeDef init;
    //Change to actual i2c declaration
    init = HAL_I2C_Master_Transmit(&hi2c1, DEPTH_SENSOR_ADDR, &resetByte, 1, DEPTH_SENSOR_TIMEOUT);
    //PROM Calibration Coefficients
    HAL_I2C_Master_Transmit(&hi2c1, DEPTH_SENSOR_ADDR, &PROM_Calibration, 1, DEPTH_SENSOR_TIMEOUT);
    HAL_I2C_Master_Receive(&hi2c1, (DEPTH_SENSOR_ADDR | 0x01), &calibration_coeff, 2, DEPTH_SENSOR_TIMEOUT);

    //Blink LED
    if (init == HAL_OK)
    {
        HAL_GPIO_TogglePin(GPIOI, GPIO_PIN_1);
    }
    else
    {
        HAL_GPIO_TogglePin(GPIOI, GPIO_PIN_1);
        HAL_Delay(1000);
        HAL_GPIO_TogglePin(GPIOI, GPIO_PIN_1);
        HAL_Delay(1000);
    }

}

void depth_sensor_task(DepthSensorMessage* output_message)
{
    const uint8_t d1_osr4096 = 0x48;
    const uint8_t d2_osr4096 = 0x58;
    const uint8_t ADC_read = 0x00;
    uint8_t uncomp_temp[3];
    uint8_t uncomp_pressure[3];
    //Read Pressure
	  //Initiate Pressure Conversion
	  HAL_I2C_Master_Transmit(&hi2c1, DEPTH_SENSOR_ADDR, &d1_osr4096, 1, HAL_MAX_DELAY);
	  HAL_Delay(100);
	  //ADC Read Command
	  HAL_I2C_Master_Transmit(&hi2c1, DEPTH_SENSOR_ADDR, &ADC_read, 1, HAL_MAX_DELAY);
	  //Sensor Answer
	  HAL_I2C_Master_Receive(&hi2c1, (DEPTH_SENSOR_ADDR | 0x01), &uncomp_pressure, 3, HAL_MAX_DELAY);

	  uint32_t pressure = uncomp_pressure[2] << 16 | uncomp_pressure[1] << 8 | uncomp_pressure[0];

      //Add to output message
      output_message -> pressure_reading = (float)pressure;


	  //Read Temp
	  //Init Temp Conversion
	  HAL_I2C_Master_Transmit(&hi2c1, DEPTH_SENSOR_ADDR, &d2_osr4096, 1, DEPTH_SENSOR_TIMEOUT);
	  HAL_Delay(100);
	  //ADC Read
	  HAL_I2C_Master_Transmit(&hi2c1, DEPTH_SENSOR_ADDR, &ADC_read, 1, DEPTH_SENSOR_TIMEOUT);
	  //Sensor Answer
	  HAL_I2C_Master_Receive(&hi2c1, (DEPTH_SENSOR_ADDR | 0x01), &uncomp_temp, 3, DEPTH_SENSOR_TIMEOUT);

	  uint32_t temp = uncomp_temp[2] << 16 | uncomp_temp[1] << 8 | uncomp_temp[0];

      //Add to output message
      output_message -> temperature_reading = (float)temp;
}
