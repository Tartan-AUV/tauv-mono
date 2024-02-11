uint8_t byteWritten;
uint8_t buf[11];
uint8_t response[11];

typedef enum { 
  TEST_CODE = 0x01,
  HEARTBEAT = 0xFF,
  SET_POWER = 0x88,
  THRUSTER_POWER = 0x44,
  THRUSTER_POWER_V = 0x45,
  GET_VOLTAGE = 0x22
} CommandType;

void setup() {
  pinMode(12, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(9, OUTPUT);
  digitalWrite(12, HIGH);
  digitalWrite(11, LOW);
  digitalWrite(10, HIGH);
  digitalWrite(9, LOW);
  Serial.begin(9600);

}

void loop() {
  if (Serial.available() > 0) {
    byteWritten = Serial.read();
    memcpy(buf, &buf[1], 10);
    buf[10] = byteWritten;

    if (buf[0] == GET_VOLTAGE) {
      response[0] = GET_VOLTAGE;
      uint32_t voltage = analogRead(A0);
      memcpy(response + 1, &voltage, sizeof(uint32_t));
      uint16_t checksum = hash(buf);
      memcpy(response + 9, &checksum, sizeof(uint16_t));

      Serial.write(response, 11);
    }
  }

}

uint16_t hash(uint8_t buf[9]) {
  uint32_t checksum = 0;
  uint32_t powfactor = 1;
  for (int i = 0; i < 9; i++) {
    // checksum += ((uint16_t)((buf[i]) * pow(256, 9-1-i)) % 65521);
    checksum = (checksum + (buf[9-1-i] * powfactor)) % 65521;
    powfactor = (powfactor * 256) % 65521;
  }
  return checksum;
}

bool checkSum(uint8_t buf[11]) {
  return hash(buf) == *((uint16_t *)(&buf[9]));
}
