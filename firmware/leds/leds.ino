
#include <Adafruit_DotStar.h>
#include <SPI.h>

#define SIDELEN 45
#define FRONTLEN 13

// Here's how to control the LEDs from any two pins:
#define DATAPIN    11
#define CLOCKPIN   13

Adafruit_DotStar strip(SIDELEN*2+FRONTLEN, DATAPIN, CLOCKPIN, DOTSTAR_BRG);


void setup() {

#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000L)
  clock_prescale_set(clock_div_1); // Enable 16 MHz on Trinket
#endif

  strip.begin(); // Initialize pins for output
  strip.show();  // Turn all LEDs off ASAP

  Serial.begin(9600);
}


void loop() {
  strip.clear();
  int t = millis();
  frontRainbow(t, 500);
  sideRace(t, 1000);
  strip.show();  // Turn all LEDs off 
}


void setSidePixelColor(int i, uint32_t color) {
  strip.setPixelColor(i, color);
  strip.setPixelColor(SIDELEN*2+FRONTLEN-1-i, color);
}
void setFrontPixelColor(int i, uint32_t color) {
  strip.setPixelColor(SIDELEN+i, color);
}
