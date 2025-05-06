/* TAUV RTVC */
/* 50Hz CAN Task */
/* Author: Victor Zayakov */

#include "can50hz.h"

#include "can.h"
#include "portmacro.h"
#include "vesc.h"
#include "logging.h"

void Task_CAN50Hz(const CAN50HzInputMessage *inputMessage, CAN50HzMessage *outputMessage) {

	const Eth50HzOutputMessage recvEthMsg = inputMessage->outputMsg;

	for (size_t i = 0; i < recvEthMsg.NumEscMsgs; i++) {
		for (size_t j = 0; j < CONF_N_ESCS; j++) {

			int32_t rpm = (recvEthMsg.OutMsgs[i]).EscRpm[j];
			bool enable = (recvEthMsg.OutMsgs[i]).EscEnable[j];

			const CAN_TxHeaderTypeDef esc_header = {
				.StdId = 0,
				.ExtId = vesc_get_can_msg_id(VESC_SET_RPM, (uint8_t) (127 + j)),
				.IDE = CAN_ID_EXT,
				.RTR = CAN_RTR_DATA,
				.DLC = 4,
				.TransmitGlobalTime = DISABLE,
			};

			// Set RPM values to 0 if ESCs not enabled
			uint8_t payload[4];
			if (!enable) {
				vesc_get_rpm_payload((int32_t)(0), payload, sizeof(payload));
			}
			else {
				vesc_get_rpm_payload(rpm, payload, sizeof(payload));
			}

			uint32_t mailbox;

			printf("Sending CAN 50Hz\n");
			HAL_CAN_AddTxMessage(&hcan1, &esc_header, payload, &mailbox);
		}
	}
}
