#include <Wire.h>
#include "MS5837.h"

#include <ros.h>
#include <std_msgs/Float32.h>

#define SENSE_MUX_EN_L 9
#define SENSE_MUX_S2 12
#define SENSE_MUX_S1 11
#define SENSE_MUX_S0 10
#define SENSE_MUX_OUT A0

#define SENSE_MUX_VCC 0
#define SENSE_MUX_V12P0 1
#define SENSE_MUX_V5P0S 2
#define SENSE_MUX_V5P0 3
#define SENSE_MUX_V3P3 4
#define SENSE_MUX_VTHRUST 5
#define SENSE_MUX_ITHRUST 6

MS5837 depth_sensor;

ros::NodeHandle nh;

std_msgs::Float32 float_msg;
ros::Publisher depth_pub("depth", &float_msg);

ros::Publisher voltage_vcc_pub("voltages/vcc", &float_msg);
ros::Publisher voltage_v12p0_pub("voltages/v12p0", &float_msg);
ros::Publisher voltage_v5p0s_pub("voltages/v5p0s", &float_msg);
ros::Publisher voltage_v5p0_pub("voltages/v5p0", &float_msg);
ros::Publisher voltage_v3p3_pub("voltages/v3p3", &float_msg);
ros::Publisher voltage_vthrust_pub("voltages/vthrust", &float_msg);
ros::Publisher voltage_ithrust_pub("voltages/ithrust", &float_msg);

void sense_mux_init() {
  pinMode(SENSE_MUX_EN_L, OUTPUT);
  digitalWrite(SENSE_MUX_EN_L, true);
  pinMode(SENSE_MUX_S2, OUTPUT);
  pinMode(SENSE_MUX_S1, OUTPUT);
  pinMode(SENSE_MUX_S0, OUTPUT);
}

float sense_mux_read_voltage(uint8_t channel) {
  digitalWrite(SENSE_MUX_S2, channel & 0b0100);
  digitalWrite(SENSE_MUX_S1, channel & 0b0010);
  digitalWrite(SENSE_MUX_S0, channel & 0b0001);
  digitalWrite(SENSE_MUX_EN_L, false);
  
  uint16_t voltage_raw = analogRead(SENSE_MUX_OUT);

  digitalWrite(SENSE_MUX_EN_L, true);

  float conversion = 3.9375;
  if (channel == SENSE_MUX_V3P3) {
    conversion = 1;
  } else if (channel == SENSE_MUX_ITHRUST) {
    conversion = 1000;
  } 

  return ((float) voltage_raw * (5.0 / 1024.0)) * conversion;
}


void setup() {
  nh.initNode();

  nh.advertise(depth_pub);
  nh.advertise(voltage_vcc_pub);
  nh.advertise(voltage_v12p0_pub);
  nh.advertise(voltage_v5p0s_pub);
  nh.advertise(voltage_v5p0_pub);
  nh.advertise(voltage_v3p3_pub);
  nh.advertise(voltage_vthrust_pub);
  nh.advertise(voltage_ithrust_pub);

  Wire.begin();

  depth_sensor.setModel(MS5837::MS5837_02BA);
  depth_sensor.setFluidDensity(997); // kg/m^3 (freshwater, 1029 for seawater)

  sense_mux_init();
}

void loop() {
  if (!depth_sensor.init()) {
    Serial.println("NaN");
  } else {
    depth_sensor.read();
    float_msg.data = depth_sensor.depth();
    depth_pub.publish(&float_msg);
  }

  float_msg.data = sense_mux_read_voltage(SENSE_MUX_VCC);
  voltage_vcc_pub.publish(&float_msg);

  float_msg.data = sense_mux_read_voltage(SENSE_MUX_V12P0);
  voltage_v12p0_pub.publish(&float_msg);

  float_msg.data = sense_mux_read_voltage(SENSE_MUX_V5P0S);
  voltage_v5p0s_pub.publish(&float_msg);

  float_msg.data = sense_mux_read_voltage(SENSE_MUX_V5P0);
  voltage_v5p0_pub.publish(&float_msg);

  float_msg.data = sense_mux_read_voltage(SENSE_MUX_V3P3);
  voltage_v3p3_pub.publish(&float_msg);

  float_msg.data = sense_mux_read_voltage(SENSE_MUX_VTHRUST);
  voltage_vthrust_pub.publish(&float_msg);

  float_msg.data = sense_mux_read_voltage(SENSE_MUX_ITHRUST);
  voltage_ithrust_pub.publish(&float_msg);

  nh.spinOnce();
}
