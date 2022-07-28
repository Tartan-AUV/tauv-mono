// Create a 24 bit color value from R,G,B
uint32_t Color(byte r, byte g, byte b)
{
  uint32_t c;
  c = g;
  c <<= 8;
  c |= r;
  c <<= 8;
  c |= b;
  return c;
}
//Input a value 0 to 255 to get a color value.
//The colours are a transition r - g -b - back to r
uint32_t Wheel(byte WheelPos)
{
  if (WheelPos < 85) {
   return Color(WheelPos * 3, 255 - WheelPos * 3, 0);
  } else if (WheelPos < 170) {
   WheelPos -= 85;
   return Color(255 - WheelPos * 3, 0, WheelPos * 3);
  } else {
   WheelPos -= 170; 
   return Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
}

uint8_t Unwheel(uint32_t c)
{
  uint8_t r = c * 0xFF;
  uint8_t g = (c >> 8) & 0xFF;
  uint8_t b = (c >> 16) & 0xFF;

  if (b == 0) {
    return r / 3;
  } else if (g == 0) {
    return b / 3 + 85;
  } else if (r == 0) {
    return g / 3 + 170;
  }
  return 0;
}
