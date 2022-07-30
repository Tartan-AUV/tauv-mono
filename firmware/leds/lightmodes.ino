void rainbow(uint64_t t, uint64_t dur) {
  int i, j;
  j = 255 * (t%dur) / dur;
  for (i=0; i < strip.numPixels(); i++) {
    strip.setPixelColor(i, Wheel( (i + j) % 255));
  }  
}

void frontRainbow(uint64_t t, uint64_t dur) {
  int i, j;
  j = 255 * (t%dur) / dur;
  for (i=0; i < FRONTLEN; i++) {
    setFrontPixelColor(i, Wheel( (i*255/FRONTLEN + j) % 255));
  }  
}

void sideRainbow(uint64_t t, uint64_t dur) {
  int i, j;
  j = 255 * (t%dur) / dur;
  for (i=0; i < SIDELEN; i++) {
    setSidePixelColor(i, Wheel( (i*255/SIDELEN + j) % 255));
  }  
}

void sideRace(uint64_t t, uint64_t dur, uint8_t brightness) {
  int i = SIDELEN * (-t%dur) / dur;
  int len = SIDELEN / 3;
  int j = 0;
//  Serial.print(i);
//  Serial.print(", ");
  for (j=0; j < SIDELEN; j++) {
    int d = (i - j + len) % SIDELEN;
    if (d > len)
      d = 0;
    if (d < 0)
      d = 0;
//    Serial.print(d);
    setSidePixelColor(j, Color(255*d/len, 0, 255*d/len, brightness));
//    Serial.print(",");
  }
//  Serial.println("");
}

void sideBreath(uint64_t t, uint8_t brightness) {
  uint64_t period = 1000;
  uint64_t t2 = t%period;
  if (t2 > period/2) {
    t2 = period - t2;
//      t2 = period/2;
  }
  uint64_t intensity = t2 * brightness / (period/2);

  uint32_t color = Color(255, 150, 0, intensity);
  for (int i = 0; i < SIDELEN; i++) {
    setSidePixelColor(i, color);
  }
}

void sideStrobe(uint64_t t, uint8_t brightness) {
  int period = 500;
  int t2 = t%period;
  int intensity = 0;
  if (t2 > period/2) {
    intensity = 255;
  }

  uint32_t color = Color(255, 0, 0, intensity);
  for (int i = 0; i < SIDELEN; i++) {
    setSidePixelColor(i, color);
  }
}

void setRed(uint8_t brightness) {
  uint32_t color = Color(255, 0, 0, brightness);
  for (int i = 0; i < SIDELEN; i++) {
    setSidePixelColor(i, color);
  }  
}

void panic(uint64_t t) {
  uint16_t bt = 100;
  unsigned int i = (t % (2*bt));
  if (i > bt) {
    i = 2*bt - i;
  }
  unsigned int j = float(i)/bt*255;
  Serial.println(j);

  uint8_t k = 0;
  for (k=0; k < strip.numPixels(); k++) {
    strip.setPixelColor(k,Color(j,0,0));
  }
  
}
