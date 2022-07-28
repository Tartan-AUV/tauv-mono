
#include <Adafruit_DotStar.h>
#include <SPI.h>
//#include <Wire.h>
#include <SoftwareWire.h>
#include "depth.h"

#define SIDELEN 45
#define FRONTLEN 13

// Here's how to control the LEDs from any two pins:
#define DATAPIN    11
#define CLOCKPIN   13

Adafruit_DotStar strip(SIDELEN*2+FRONTLEN, DATAPIN, CLOCKPIN, DOTSTAR_BRG);
MS5837 sensor;
SoftwareWire wire(A4, A5, false);

void setup() {

#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000L)
  clock_prescale_set(clock_div_1); // Enable 16 MHz on Trinket
#endif

  wire.setClock(10000);
  wire.begin();
  Serial.begin(115200);

  strip.begin(); // Initialize pins for output
  strip.show();  // Turn all LEDs off ASAP
}


uint64_t last_depth_read = 0;
bool sensor_init = false;

void loop() {
  int t = millis();

  // get the state
//  if (Serial.readln()

  
  // command the strip
  strip.clear();
  sideRace(t, 1000, 25);
  strip.show();

  
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


void setSidePixelColor(int i, uint32_t color) {
  strip.setPixelColor(i, color);
  strip.setPixelColor(SIDELEN*2+FRONTLEN-1-i, color);
}
void setFrontPixelColor(int i, uint32_t color) {
  strip.setPixelColor(SIDELEN+i, color);
}
