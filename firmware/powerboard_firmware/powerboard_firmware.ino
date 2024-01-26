size_t byteWritten = 0; // for incoming serial data

// uint8_t all[11] = {0, 0, 0, 0, 0, 0, 0, 255, 25};
uint8_t all[11] = {0, 0, 0, 0, 0, 0, 1, 255, 25};
int voltage_pin = A0;

typedef enum { 
  GET_VOLTAGE = 0x22
} CommandType;

void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}

struct RequestType {
  char command_id;
  union {
    // payload
  };
};

struct ResponseType {
  char command_id;
  union {
    int voltage;
  };
};

void loop() {
  // send data only when you receive data:
  if (Serial.available() > 0) {
    // read the incoming byte:
    struct RequestType testing;
    uint8_t encoded_request[11];
    memcpy(encoded_request, &GET_VOLTAGE, sizeof(CommandType));
    memcpy(encoded_request[9], &hash(encoded_request), sizeof(uint16_t));

    testing = parse(encoded_request);
    Serial.println(testing.command_id, DEC);

    uint8_t *encoded_response = createResponse(testing);

    // encoded_request[0] = GET_VOLTAGE;
    

    char buf[11];
    byteWritten = Serial.readBytes(buf, 11);
    uint16_t test = hash(all);

    // say what you got:
    Serial.print("I received: ");
    Serial.println(byteWritten, DEC);
    Serial.println((char *)all);
    printBytes(all, 11);
    Serial.println(test, DEC);
  }
}

// hash function
// @param buf 9 bytes char array with encoded message from jetson
// @return 2 bytes computed from rabin fingerprint
uint16_t hash(uint8_t buf[9]) {
  uint32_t checksum = 0;
  uint32_t powfactor = 1;
  for (int i = 0; i < 9; i++) {
    // checksum += ((uint16_t)((buf[i]) * pow(256, 9-1-i)) % 65521);
    checksum = (checksum + (buf[9-1-i] * powfactor)) % 65521;
    Serial.println(checksum, DEC);
    powfactor = (powfactor * 256) % 65521;
  }
  return checksum;
}

struct RequestType *parse(uint8_t buf[11]) {
  uint8_t buffer[9];
  uint8_t response[11];
  memcpy(buffer, buf, sizeof(uint8_t) * 9);

  uint16_t sum = hash(buffer);
  uint16_t checksum;
  memcpy(&checksum, buf+9, sizeof(uint16_t));

  if (sum != checksum) {
    Serial.println(sum, DEC);
    Serial.println(checksum, DEC);
    return NULL;
  }

  struct RequestType request;

  switch (buf[0]) {
    case GET_VOLTAGE: {
      request.command_id = GET_VOLTAGE;
    }
  }

  return &request;
}

uint8_t *createResponse(struct RequestType request) {
  struct ResponseType response;
  uint8_t response_array[11];
  uint16_t checksum;

  switch (request.command_id) {
    case GET_VOLTAGE: {
      response.command_id = GET_VOLTAGE;
      response.voltage = analogRead(voltage_pin);

      memcpy(response_array, &response, sizeof(response));
      checksum = hash(response_array);
      memcpy(response_array + 9, &checksum, sizeof(uint16_t));
    }
  }

  return response_array;
}

void printBytes(char *str, size_t length) {
  for (int i = 0; i < length; i++) {
    Serial.print(str[i], DEC);
  }
  Serial.println();
}

