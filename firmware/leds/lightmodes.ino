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

void sideRace(uint64_t t, uint64_t dur) {
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
    setSidePixelColor(j, Color(255*d/len, 0, 255*d/len));
//    Serial.print(",");
  }
//  Serial.println("");
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
