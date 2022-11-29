#include "depth.h"
#define DATAPIN    11
#define CLOCKPIN   13

MS5837 sensor;

uint64_t last_depth_read = 0;
bool sensor_init = false;

SoftwareWire wire(A4, A5, false);

void setup() {
  wire.setClock(20000);
  wire.begin();
  Serial.begin(115200);
}

void loop() {
  // depth telemetry
  if (millis() - last_depth_read > 100) {
    last_depth_read = millis();
    
    if (!sensor_init) {
      sensor_init = sensor.init(wire);
    } else {
      sensor_init = sensor.read();
    }
    
    Serial.print("D,");
    if (sensor_init) {
      Serial.println(sensor.depth()); 
    } else {
      Serial.println("NaN");
    }
  }
}
