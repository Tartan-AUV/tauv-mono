size_t byteWritten = 0; // for incoming serial data
int voltage_pin = A0;
uint8_t buf[11];


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
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}

struct RequestType {
  uint8_t command_id;
  uint8_t payload[8];
  uint16_t sumHash; 
};

struct ResponseType {
  uint8_t command_id;
  uint8_t payload[8];
  uint16_t sumHash;
};

void loop() {
  // send data only when you receive data:
  if (Serial.available() > 0) {
    // read the incoming byte:
    // struct RequestType testing;
    //uint8_t encoded_request[11];
    // memcpy(encoded_request, &GET_VOLTAGE, sizeof(CommandType));
    // memcpy(encoded_request[9], &hash(encoded_request), sizeof(uint16_t));

    // testing = parse(encoded_request);
    // Serial.println(testing.command_id, DEC);

    // uint8_t *encoded_response = createResponse(testing);

    // encoded_request[0] = GET_VOLTAGE;
    
    //byteWritten = Serial.readBytes(buf, 11);
    //uint16_t test = hash(all);

    // say what you got:
    //Serial.print("I received: ");
    //Serial.println(byteWritten, DEC);
    //Serial.println((char *)all);
    //printBytes(all, 11);
    //Serial.println(test, DEC);

    byteWritten = Serial.read();
    
    memcpy(buf, &buf[1], 10 * sizeof(uint8_t));
    buf[10] = (uint8_t) byteWritten;

    if (checkSum(buf)) {
      struct RequestType *parsedRequest = parse(buf);
      uint8_t *response = createResponse(*parsedRequest);

      Serial.write(response, 11);
    }
  }
}

/** hash function
 * @param buf 9 bytes char array with encoded message from jetson
 * @return 2 bytes computed from rabin fingerprint
 */
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

struct RequestType *parse(uint8_t buf[11]) {

  struct RequestType request;

  switch (buf[0]) {
    case GET_VOLTAGE:
      request.command_id = GET_VOLTAGE;
      break;
    case HEARTBEAT:
      request.command_id = HEARTBEAT;
      break;
    case SET_POWER:
      request.command_id = SET_POWER;
      break;
    case THRUSTER_POWER:
      request.command_id = THRUSTER_POWER;
      request.payload[0] = buf[1];
      break;
    case THRUSTER_POWER_V:
      request.command_id = THRUSTER_POWER_V;
      request.payload[0] = buf[1];
      break;
  }

  return &request;
}

uint8_t *createResponse(struct RequestType request) {
  struct ResponseType response;
  uint8_t response_array[11];
  uint16_t checksum;
  
  memset(response_array, 0x00, 11);

  switch (request.command_id) {
    case GET_VOLTAGE:
      response.command_id = GET_VOLTAGE;
      response.payload[0] = analogRead(voltage_pin);
      memcpy(response_array, &response,9);
      break;
    
    case HEARTBEAT:
      response_array[0] = HEARTBEAT;
      memset(&response_array[1], 0xFF, 8);
      break;
    
    case SET_POWER:
      response.command_id = SET_POWER;
      // TBD implementation
      memcpy(response_array, &response, 9);
      break;
    
    case THRUSTER_POWER:
      response.command_id = THRUSTER_POWER;
      uint8_t thruster_bool = request.payload[0];
      if (thruster_bool == 0xFF) {
        // Turn on thruster power
      }
      else if (thruster_bool == 0x00) {
        // Turn of thruster power
      }
      memset(response_array, THRUSTER_POWER, 1);
      break;
    
    case THRUSTER_POWER_V:
      response.command_id = THRUSTER_POWER_V;
      uint8_t thruster_v_bool = request.payload[0];
      if (thruster_v_bool == 0xFF) {
        // Turn on thruster power
      }
      else if (thruster_v_bool == 0x00) {
        // Turn of thruster power
      }
      memset(response_array, THRUSTER_POWER, 1);
      break;
  }

  checksum = hash(response_array);
  memcpy(response_array + 9, &checksum, sizeof(uint16_t));

  return response_array;
}