/* TAUV RTVC */
/* 100Hz Ethernet Task */
/* Author: Gleb Ryabtsev */

/* INCLUDES */
#ifndef DEPTHSENSOR_H
#define DEPTHSENSOR_H

#include "messages.h"
#include <stdint.h>

/* MESSAGE DEFINITIONS */


#define DEPTH_SENSOR_TIMEOUT (50)
#define DEPTH_SENSOR_ADDR (0x76 << 1)


/* TASK DECLARATION */
void depth_sensor_init();
void depth_sensor_task(DepthSensorMessage* output_message);

#endif // DEPTHSENSOR_H
