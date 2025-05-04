#include "can100hz.h"

#include <assert.h>

#include "main.h"
#include "messages.h"
#include "vehicle_config.h"

#include "logging.h"

#include "can.h"
#include "portmacro.h"
#include "projdefs.h"

#include "vesc.h"
#include "xsens.h"

#define XSENS_RECVD_SAMPLE_TIME 0x01
#define XSENS_RECVD_ORIENTATION_QUATERNION 0x02
#define XSENS_RECVD_RATE_OF_TURN 0x04
#define XSENS_RECVD_FREE_ACCELERATION 0x08
#define XSENS_FRAME_COMPLETE                                                   \
  (XSENS_RECVD_SAMPLE_TIME | XSENS_RECVD_ORIENTATION_QUATERNION |              \
   XSENS_RECVD_RATE_OF_TURN | XSENS_RECVD_FREE_ACCELERATION)

static XsensIMUFrame currXsensIMUFrame;
static uint8_t xsensImuFrameStatus;

void Task_CAN100Hz_Init() { xsensImuFrameStatus = 0; }

void Task_CAN100Hz(const CAN100HzInputMessage *inputMsg,
                   CAN100HzMessage *outputMsg) {
  const uint32_t rpm = 10000;
  HAL_StatusTypeDef status;
  /* ESC TX */
  for (size_t i = 0; i < 1; ++i) {
    //        const CAN_TxHeaderTypeDef esc_header = {
    //            .StdId = 0,
    //            .ExtId = vesc_get_can_msg_id(VESC_SET_RPM, (uint8_t) 79),
    //            .IDE = CAN_ID_EXT,
    //            .RTR = CAN_RTR_DATA,
    //            .DLC = 4,
    //            .TransmitGlobalTime = DISABLE,
    //        };
    //
    //        uint8_t payload[4];
    //        vesc_get_rpm_payload(rpm, payload, sizeof(payload));
    //
    //        uint32_t mailbox;
    //
    //        HAL_CAN_AddTxMessage(&hcan1, &esc_header, payload, &mailbox);
  }

  // IMU messages
  size_t currImuMsgIdx = 0;
  XsensIMUFrame *imuFrames = outputMsg->ImuFrames;
  if (xsensImuFrameStatus != 0) {
    imuFrames[0] = currXsensIMUFrame;
  }

  CANRxMessage_t canRxMsg;
  size_t canMsgCounter = 0;
  while (xQueueReceive(can100HzRxQueue, &canRxMsg, 0) == pdTRUE) {
    const uint32_t canMsgId = canRxMsg.header.ExtId;

    if (canMsgId >= 0x80) {
      // VESC message
    } else {
      // IMU message
      switch (canMsgId) {
      case CAN_MSG_ID_XSENS_SAMPLE_TIME:
        // DBG("Xsens SampleTime");
        if (xsensImuFrameStatus != 0) {
          // previous frame is incomplete, need to ignore
          xsensImuFrameStatus = 0;
          WARN("IMU Frame Error (SmpleTime)!");
        }
        xsensImuFrameStatus |= XSENS_RECVD_SAMPLE_TIME;
        imuFrames[currImuMsgIdx].SampleTime =
            CAN_MsgParse_Xsens_SampleTime(canRxMsg.data);
        break;
      case CAN_MSG_ID_XSENS_ORIENTATION_QUATERNION:
        // DBG("Xsens OrientationQuat");
        if (xsensImuFrameStatus & XSENS_RECVD_ORIENTATION_QUATERNION) {
          WARN("IMU Frame Error (OrientationQuat)!");
          break;
        }
        xsensImuFrameStatus |= XSENS_RECVD_ORIENTATION_QUATERNION;
        imuFrames[currImuMsgIdx].Orientation =
            CAN_MsgParse_Xsens_Orientation(canRxMsg.data);
        break;
      case CAN_MSG_ID_XSENS_RATE_OF_TURN:
        // DBG("Xsens RateOfTurn");
        if (xsensImuFrameStatus & XSENS_RECVD_RATE_OF_TURN) {
          WARN("IMU Frame Error (RateOfTurn)!");
          break;
        }
        xsensImuFrameStatus |= XSENS_RECVD_RATE_OF_TURN;
        imuFrames[currImuMsgIdx].RateOfTurn =
            CAN_MsgParse_Xsens_RateOfTurn(canRxMsg.data);
        break;
      case CAN_MSG_ID_XSENS_FREE_ACCELERATION:
        if (xsensImuFrameStatus & XSENS_RECVD_FREE_ACCELERATION) {
          WARN("IMU Frame Error (FreeAcceleration)!");
          break;
        }
        xsensImuFrameStatus |= XSENS_RECVD_FREE_ACCELERATION;
        imuFrames[currImuMsgIdx].FreeAcceleration =
            CAN_MsgParse_Xsens_FreeAcceleration(canRxMsg.data);
        break;
      default:
    	WARN("CAN msg id unknown: %#4x", canMsgId);
      }
      ++canMsgCounter;
      assert(currImuMsgIdx < CAN_XSENS_IMU_FRAMES_MAX);
      if (xsensImuFrameStatus == XSENS_FRAME_COMPLETE) {
        ++currImuMsgIdx;
        xsensImuFrameStatus = 0;
      }
    }
  }

  if (xsensImuFrameStatus != 0) {
    currXsensIMUFrame = imuFrames[currImuMsgIdx];
  }

  outputMsg->NImuFrames = currImuMsgIdx; // exclude the last frame which either
                                         // empty or partially complete
}
