/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      Implementation for Ethernet 100Hz module sending IMU data to Jetson
 *
 *****************************************************************************/

#include "Eth100HzModule.hpp"

#include "Logging.hpp"
#include "eth_msg_rtvc_jetson_100_generated.h"
#include "lwip/inet.h"

extern ip_addr_t jetsonAddr;

using namespace TAUV;

ModuleInitResult Eth100HzModule::init() {
  sock_.init();
  sock_.bind(Config::Network::jetson_100hz_port);
  
  LOG_INFO("Eth100HzModule: Initialized UDP socket on port %d", Config::Network::jetson_100hz_port);
  return ModuleInitResult::OK;
}

ModuleRunResult Eth100HzModule::run() {
  const MTI300Message& mti300Msg = input_interface_.getMTI300Message();
  
  // If there are no valid MTI300 messages, skip transmission
  if (mti300Msg.count == 0) {
    return ModuleRunResult::OK;
  }

  flatbuffers::FlatBufferBuilder fbb{1024};

  std::array<flatbuffers::Offset<TAUV_FB::XsensIMUFrame>, Config::IMU::queueLength> fbImuFrames;

  assert(mti300Msg.count <= Config::IMU::queueLength);

  for (int i = 0; i < mti300Msg.count; i++) {
    const auto& msg = mti300Msg.frames[i];
    
    // Skip frames that don't have necessary data
    if (!msg.quaternion.has_value() || !msg.angularVelocity.has_value() || 
        !msg.freeAcceleration.has_value()) {
      continue;
    }
    
    // Create orientation quaternion
    TAUV_FB::Quat orientation(
      msg.quaternion.value()[0],
      msg.quaternion.value()[1],
      msg.quaternion.value()[2],
      msg.quaternion.value()[3]
    );
    
    // Create rate of turn vector
    TAUV_FB::Vec3 rateOfTurn(
      msg.angularVelocity.value()[0],
      msg.angularVelocity.value()[1],
      msg.angularVelocity.value()[2]
    );
    
    // Create free acceleration vector
    TAUV_FB::Vec3 freeAcceleration(
      msg.freeAcceleration.value()[0],
      msg.freeAcceleration.value()[1],
      msg.freeAcceleration.value()[2]
    );
    
    // Create the IMU frame and add it directly to the vector
    TAUV_FB::XsensIMUFrameBuilder fbImuFrameBuilder(fbb);
    if (msg.sampleTimeFine.has_value())
        fbImuFrameBuilder.add_sample_time_fine(msg.sampleTimeFine.value());
    if (msg.packetCounter.has_value())
      fbImuFrameBuilder.add_packet_counter(msg.packetCounter.value());
    if (msg.quaternion.has_value())
      fbImuFrameBuilder.add_orientation(&orientation);
    if (msg.angularVelocity.has_value())
      fbImuFrameBuilder.add_rate_of_turn(&rateOfTurn);
    if (msg.freeAcceleration.has_value())
      fbImuFrameBuilder.add_free_acceleration(&freeAcceleration);
    if (msg.pressure.has_value())
      fbImuFrameBuilder.add_pressure(msg.pressure.value());
    if (msg.temperature.has_value())
      fbImuFrameBuilder.add_temperature(msg.temperature.value());
    fbImuFrames[i] = fbImuFrameBuilder.Finish();
  }
  fbb.StartVector(sizeof(flatbuffers::Offset<TAUV_FB::XsensIMUFrame>),
                     mti300Msg.count,
                     sizeof(flatbuffers::Offset<TAUV_FB::XsensIMUFrame>));


  // Use non-object-based API to build the vector
  for (size_t i = 0; i < mti300Msg.count; i++)
      fbb.PushElement(fbImuFrames[i]);

  // End the vector
  auto fbImuFramesVector = fbb.EndVector(mti300Msg.count);
  
  // Create the root Eth100HzMsg
  auto fbEthMsg = TAUV_FB::CreateEth100HzMsg(fbb, fbImuFramesVector);
  
  // Finish the buffer
  fbb.Finish(fbEthMsg);
  
  // Get the buffer and its size
  uint8_t* buffer = fbb.GetBufferPointer();
  int size = fbb.GetSize();
  
  // Send the message to the Jetson
  sock_.send(jetsonAddr, Config::Network::jetson_100hz_port, buffer, size);
  
  return ModuleRunResult::OK;
}
