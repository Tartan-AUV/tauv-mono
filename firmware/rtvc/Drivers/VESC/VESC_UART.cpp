#include "VESC_UART.hpp"

#include <stdint.h>

#include <cstddef>
#include <cstring>

extern "C" {
#include "stm32f7xx_hal.h"
}

namespace TAUV::VESC {

VESC_UART::VESC_UART(uint32_t timeout_ms) : _TIMEOUT(timeout_ms) {
  nunchuck.valueX = 127;
  nunchuck.valueY = 127;
  nunchuck.lowerButton = false;
  nunchuck.upperButton = false;
}

int VESC_UART::receiveUartMessage(uint8_t* payloadReceived) {
  if (huart == nullptr) return -1;  // huart is a pointer to HAL UART handle

  uint16_t counter = 0;
  uint16_t endMessage = 256;
  bool messageRead = false;
  uint8_t messageReceived[256];
  uint16_t lenPayload = 0;

  uint32_t timeout = HAL_GetTick() + _TIMEOUT;

  while (HAL_GetTick() < timeout && !messageRead) {
    uint8_t byte = 0;
    if (HAL_UART_Receive(huart, &byte, 1, 1) == HAL_OK) {  // 1 ms polling timeout
      messageReceived[counter++] = byte;

      if (counter == 2) {
        switch (messageReceived[0]) {
          case 2:
            lenPayload = messageReceived[1];
            endMessage = lenPayload + 5;
            break;

          case 3:
            // if (debugPort) {
            //   debugPort->println("Message is larger than 256 bytes - not supported");
            // }
            return 0;

          default:
            // if (debugPort) {
            //   debugPort->println("Invalid start byte");
            // }
            return 0;
        }
      }

      if (counter >= sizeof(messageReceived)) {
        break;  // prevent buffer overrun
      }

      if (counter == endMessage && messageReceived[endMessage - 1] == 3) {
        messageRead = true;
        // if (debugPort) {
        //   debugPort->println("End of message reached!");
        // }
        break;
      }
    }
  }

  if (!messageRead) {
    // if (debugPort) debugPort->println("Timeout");
    return 0;
  }

  bool unpacked = unpackPayload(messageReceived, endMessage, payloadReceived);
  return unpacked ? lenPayload : 0;
}

bool VESC_UART::unpackPayload(uint8_t* message, int lenMes, uint8_t* payload) {
  uint16_t crcMessage = 0;
  uint16_t crcPayload = 0;

  // Rebuild crc:
  crcMessage = message[lenMes - 3] << 8;
  crcMessage &= 0xFF00;
  crcMessage += message[lenMes - 2];

  // if (debugPort != NULL) {
  //   debugPort->print("SRC received: ");
  //   debugPort->println(crcMessage);
  // }

  // Extract payload:
  std::memcpy(payload, &message[2], message[1]);

  crcPayload = crc16(payload, message[1]);

  // if (debugPort != NULL) {
  //   debugPort->print("SRC calc: ");
  //   debugPort->println(crcPayload);
  // }

  if (crcPayload == crcMessage) {
    // if (debugPort != NULL) {
    //   debugPort->print("Received: ");
    //   serialPrint(message, lenMes);
    //   debugPort->println();
    //
    //   debugPort->print("Payload :      ");
    //   serialPrint(payload, message[1] - 1);
    //   debugPort->println();
    // }

    return true;
  } else {
    return false;
  }
}

int VESC_UART::packSendPayload(uint8_t* payload, int lenPay) {
  uint16_t crcPayload = crc16(payload, lenPay);
  int count = 0;
  uint8_t messageSend[256];

  if (lenPay <= 256) {
    messageSend[count++] = 2;
    messageSend[count++] = static_cast<uint8_t>(lenPay);
  } else {
    messageSend[count++] = 3;
    messageSend[count++] = static_cast<uint8_t>(lenPay >> 8);
    messageSend[count++] = static_cast<uint8_t>(lenPay & 0xFF);
  }

  std::memcpy(messageSend + count, payload, lenPay);
  count += lenPay;

  messageSend[count++] = static_cast<uint8_t>(crcPayload >> 8);
  messageSend[count++] = static_cast<uint8_t>(crcPayload & 0xFF);
  messageSend[count++] = 3;

  // if (debugPort != nullptr) {
  //   debugPort->print("Package to send: ");
  //   serialPrint(messageSend, count);
  // }
  //
  if (huart != nullptr) {
    HAL_UART_Transmit(huart, messageSend, count, HAL_MAX_DELAY);  // blocking send
  }

  return count;
}

bool VESC_UART::processReadPacket(uint8_t* message) {
  COMM_PACKET_ID packetId;
  int32_t index = 0;

  packetId = (COMM_PACKET_ID)message[0];
  message++;  // Removes the packetId from the actual message (payload)

  switch (packetId) {
    case COMM_FW_VERSION:  // Structure defined here:
                           // https://github.com/vedderb/bldc/blob/43c3bbaf91f5052a35b75c2ff17b5fe99fad94d1/commands.c#L164

      fw_version.major = message[index++];
      fw_version.minor = message[index++];
      return true;
    case COMM_GET_VALUES:  // Structure defined here:
                           // https://github.com/vedderb/bldc/blob/43c3bbaf91f5052a35b75c2ff17b5fe99fad94d1/commands.c#L164

      data.tempMosfet = buffer_get_float16(
          message, 10.0, &index);  // 2 bytes - mc_interface_temp_fet_filtered()
      data.tempMotor = buffer_get_float16(
          message, 10.0,
          &index);  // 2 bytes - mc_interface_temp_motor_filtered()
      data.avgMotorCurrent = buffer_get_float32(
          message, 100.0,
          &index);  // 4 bytes - mc_interface_read_reset_avg_motor_current()
      data.avgInputCurrent = buffer_get_float32(
          message, 100.0,
          &index);  // 4 bytes - mc_interface_read_reset_avg_input_current()
      index += 4;   // Skip 4 bytes - mc_interface_read_reset_avg_id()
      index += 4;   // Skip 4 bytes - mc_interface_read_reset_avg_iq()
      data.dutyCycleNow = buffer_get_float16(
          message, 1000.0,
          &index);  // 2 bytes - mc_interface_get_duty_cycle_now()
      data.rpm = buffer_get_float32(
          message, 1.0, &index);  // 4 bytes - mc_interface_get_rpm()
      data.inpVoltage = buffer_get_float16(
          message, 10.0, &index);  // 2 bytes - GET_INPUT_VOLTAGE()
      data.ampHours = buffer_get_float32(
          message, 10000.0,
          &index);  // 4 bytes - mc_interface_get_amp_hours(false)
      data.ampHoursCharged = buffer_get_float32(
          message, 10000.0,
          &index);  // 4 bytes - mc_interface_get_amp_hours_charged(false)
      data.wattHours = buffer_get_float32(
          message, 10000.0,
          &index);  // 4 bytes - mc_interface_get_watt_hours(false)
      data.wattHoursCharged = buffer_get_float32(
          message, 10000.0,
          &index);  // 4 bytes - mc_interface_get_watt_hours_charged(false)
      data.tachometer = buffer_get_int32(
          message,
          &index);  // 4 bytes - mc_interface_get_tachometer_value(false)
      data.tachometerAbs = buffer_get_int32(
          message,
          &index);  // 4 bytes - mc_interface_get_tachometer_abs_value(false)
      data.error = (mc_fault_code)
          message[index++];  // 1 byte  - mc_interface_get_fault()
      data.pidPos = buffer_get_float32(
          message, 1000000.0,
          &index);  // 4 bytes - mc_interface_get_pid_pos_now()
      data.id =
          message[index++];  // 1 byte  - app_get_configuration()->controller_id

      return true;

      break;

      /* case COMM_GET_VALUES_SELECTIVE:

              uint32_t mask = 0xFFFFFFFF; */

    default:
      return false;
      break;
  }
}

bool VESC_UART::getFWversion(void) { return getFWversion(0); }

bool VESC_UART::getFWversion(uint8_t canId) {
  int32_t index = 0;
  int payloadSize = (canId == 0 ? 1 : 3);
  uint8_t payload[payloadSize];

  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }
  payload[index++] = {COMM_FW_VERSION};

  packSendPayload(payload, payloadSize);

  uint8_t message[256];
  int messageLength = receiveUartMessage(message);
  if (messageLength > 0) {
    return processReadPacket(message);
  }
  return false;
}

bool VESC_UART::getVescValues(void) { return getVescValues(0); }

bool VESC_UART::getVescValues(uint8_t canId) {
  // if (debugPort != NULL) {
  //   debugPort->println("Command: COMM_GET_VALUES " + std::string(canId));
  // }

  int32_t index = 0;
  int payloadSize = (canId == 0 ? 1 : 3);
  uint8_t payload[payloadSize];
  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }
  payload[index++] = {COMM_GET_VALUES};

  packSendPayload(payload, payloadSize);

  uint8_t message[256];
  int messageLength = receiveUartMessage(message);

  if (messageLength > 55) {
    return processReadPacket(message);
  }
  return false;
}
void VESC_UART::setNunchuckValues() { return setNunchuckValues(0); }

void VESC_UART::setNunchuckValues(uint8_t canId) {
  // if (debugPort != NULL) {
  //   debugPort->println("Command: COMM_SET_CHUCK_DATA " + std::string(canId));
  // }
  int32_t index = 0;
  int payloadSize = (canId == 0 ? 11 : 13);
  uint8_t payload[payloadSize];

  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }
  payload[index++] = {COMM_SET_CHUCK_DATA};
  payload[index++] = nunchuck.valueX;
  payload[index++] = nunchuck.valueY;
  buffer_append_bool(payload, nunchuck.lowerButton, &index);
  buffer_append_bool(payload, nunchuck.upperButton, &index);

  // Acceleration Data. Not used, Int16 (2 byte)
  payload[index++] = 0;
  payload[index++] = 0;
  payload[index++] = 0;
  payload[index++] = 0;
  payload[index++] = 0;
  payload[index++] = 0;

  // if (debugPort != NULL) {
  //   debugPort->println("Nunchuck Values:");
  //   debugPort->print("x=");
  //   debugPort->print(nunchuck.valueX);
  //   debugPort->print(" y=");
  //   debugPort->print(nunchuck.valueY);
  //   debugPort->print(" LBTN=");
  //   debugPort->print(nunchuck.lowerButton);
  //   debugPort->print(" UBTN=");
  //   debugPort->println(nunchuck.upperButton);
  // }

  packSendPayload(payload, payloadSize);
}

void VESC_UART::setCurrent(float current) { return setCurrent(current, 0); }

void VESC_UART::setCurrent(float current, uint8_t canId) {
  int32_t index = 0;
  int payloadSize = (canId == 0 ? 5 : 7);
  uint8_t payload[payloadSize];
  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }
  payload[index++] = {COMM_SET_CURRENT};
  buffer_append_int32(payload, (int32_t)(current * 1000), &index);
  packSendPayload(payload, payloadSize);
}

void VESC_UART::setBrakeCurrent(float brakeCurrent) {
  return setBrakeCurrent(brakeCurrent, 0);
}

void VESC_UART::setBrakeCurrent(float brakeCurrent, uint8_t canId) {
  int32_t index = 0;
  int payloadSize = (canId == 0 ? 5 : 7);
  uint8_t payload[payloadSize];
  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }

  payload[index++] = {COMM_SET_CURRENT_BRAKE};
  buffer_append_int32(payload, (int32_t)(brakeCurrent * 1000), &index);

  packSendPayload(payload, payloadSize);
}

void VESC_UART::setRPM(float rpm) { return setRPM(rpm, 0); }

void VESC_UART::setRPM(float rpm, uint8_t canId) {
  int32_t index = 0;
  int payloadSize = (canId == 0 ? 5 : 7);
  uint8_t payload[payloadSize];
  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }
  payload[index++] = {COMM_SET_RPM};
  buffer_append_int32(payload, (int32_t)(rpm), &index);
  packSendPayload(payload, payloadSize);
}

void VESC_UART::setDuty(float duty) { return setDuty(duty, 0); }

void VESC_UART::setDuty(float duty, uint8_t canId) {
  int32_t index = 0;
  int payloadSize = (canId == 0 ? 5 : 7);
  uint8_t payload[payloadSize];
  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }
  payload[index++] = {COMM_SET_DUTY};
  buffer_append_int32(payload, (int32_t)(duty * 100000), &index);

  packSendPayload(payload, payloadSize);
}

void VESC_UART::sendKeepalive(void) { return sendKeepalive(0); }

void VESC_UART::sendKeepalive(uint8_t canId) {
  int32_t index = 0;
  int payloadSize = (canId == 0 ? 1 : 3);
  uint8_t payload[payloadSize];
  if (canId != 0) {
    payload[index++] = {COMM_FORWARD_CAN};
    payload[index++] = canId;
  }
  payload[index++] = {COMM_ALIVE};
  packSendPayload(payload, payloadSize);
}

void VESC_UART::serialPrint(uint8_t* data, int len) {
  // if (debugPort != NULL) {
  //   for (int i = 0; i <= len; i++) {
  //     debugPort->print(data[i]);
  //     debugPort->print(" ");
  //   }
  //   debugPort->println("");
  // }
}

void VESC_UART::printVescValues() {
  // if (debugPort != NULL) {
  //   debugPort->print("avgMotorCurrent: ");
  //   debugPort->println(data.avgMotorCurrent);
  //   debugPort->print("avgInputCurrent: ");
  //   debugPort->println(data.avgInputCurrent);
  //   debugPort->print("dutyCycleNow: ");
  //   debugPort->println(data.dutyCycleNow);
  //   debugPort->print("rpm: ");
  //   debugPort->println(data.rpm);
  //   debugPort->print("inputVoltage: ");
  //   debugPort->println(data.inpVoltage);
  //   debugPort->print("ampHours: ");
  //   debugPort->println(data.ampHours);
  //   debugPort->print("ampHoursCharged: ");
  //   debugPort->println(data.ampHoursCharged);
  //   debugPort->print("wattHours: ");
  //   debugPort->println(data.wattHours);
  //   debugPort->print("wattHoursCharged: ");
  //   debugPort->println(data.wattHoursCharged);
  //   debugPort->print("tachometer: ");
  //   debugPort->println(data.tachometer);
  //   debugPort->print("tachometerAbs: ");
  //   debugPort->println(data.tachometerAbs);
  //   debugPort->print("tempMosfet: ");
  //   debugPort->println(data.tempMosfet);
  //   debugPort->print("tempMotor: ");
  //   debugPort->println(data.tempMotor);
  //   debugPort->print("error: ");
  //   debugPort->println(data.error);
  // }
}

}
