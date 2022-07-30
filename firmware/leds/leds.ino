
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

  wire.setClock(20000);
  wire.begin();
  Serial.begin(115200);

  strip.begin(); // Initialize pins for output
  strip.show();  // Turn all LEDs off ASAP
}


uint64_t last_depth_read = 0;
bool sensor_init = false;
char state = 'x';
uint64_t last_state_read = 0;

void loop() {
  int t = millis();

  // get the state
  if (Serial.available()) {
    state = Serial.read();
    last_state_read = millis();
  }

  if (millis() - last_state_read > 1000) {
    state = 'x';
  }

  // command the strip
  strip.clear();
  if (state == 'x') {
    setRed(60);
  } else if (state == 'p') {
    sideRace(t, 1000, 100);
  } else if (state == 'g') {
    sideRainbow(t, 1000);
  } else if (state == 'm') {
    sideBreath(t, 255);
  } else if (state == 'c') {
    sideStrobe(t,255);
  }

  
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
