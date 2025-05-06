/* TAUV_FB TAUV */
/* 50Hz Ethernet (Receive) Task */
/* Author: Victor Zayakov */

#include "eth50hz.h"

#include "eth_msg_jetson_rtvc_builder.h"
#include "eth_msg_jetson_rtvc_reader.h"
#include "eth_msg_jetson_rtvc_verifier.h"
#include "udp.h"
#include "vehicle_config.h"

#define RX_BUFFER_SIZE_BYTES 128

static struct udp_pcb udpPcb50hz;

/* Initialize a receive queue for Eth 50Hz */
QueueHandle_t eth50HzRxQueue;

static StaticQueue_t eth50HzRxQueueStruct;
static uint8_t eth50HzRxQueueBuffer[ETH_50HZ_QUEUE_LENGTH * sizeof(CANRxMessage_t)];

void Eth50Hz_RxQueueInit()
{

    eth50HzRxQueue = xQueueCreateStatic(ETH_50HZ_QUEUE_LENGTH,
    				   	   	   	         sizeof(Eth50HzMessage),
										 eth50HzRxQueueBuffer,
										 &eth50HzRxQueueStruct);
}

/* Task and callback definitions */

void Eth50Hz_Callback(void *arg,
					  struct udp_pcb *pcb,
					  struct pbuf *p,
					  const ip_addr_t *addr,
					  u16_t port) {

	(void)arg;
	(void)pcb;
	(void)addr;
	(void)port;

	uint8_t *buf = (uint8_t *)p->payload;
	size_t   len = p->len;

	/* Verify flatbuffer */
	if (!TAUV_FB_Eth50HzTxMsg_verify_as_root(buf, len)) {
		// TODO: log error
		pbuf_free(p);
		return;
	}

    // Create aligned buffer
    uint8_t buf_aligned[RX_BUFFER_SIZE_BYTES] __attribute__((aligned(4)));
    if (len <= RX_BUFFER_SIZE_BYTES) {
    	memcpy(buf_aligned, buf, len);
    } else {
    	// TODO: log error
    	return;
    }

    // 2) Parse root and get nested ThrusterCommand
    TAUV_FB_Eth50HzTxMsg_table_t msg = TAUV_FB_Eth50HzTxMsg_as_root(buf_aligned);
    TAUV_FB_ThrusterCommand_struct_t cmd = TAUV_FB_Eth50HzTxMsg_thruster_command(msg);

    Eth50HzMessage queue_msg;

    // 3) Pull out rpm and enabled arrays, clamp to MAX_THRUSTERS
    for (size_t i = 0; i < CONF_N_ESCS; ++i) {
        queue_msg.EscRpm   [i] = TAUV_FB_ThrusterCommand_rpm_get(cmd, i);
        queue_msg.EscEnable[i] = TAUV_FB_ThrusterCommand_enabled_get(cmd, i);
    }

    // Send received message to back of receive queue
    xQueueSendToBackFromISR(eth50HzRxQueue, &queue_msg, NULL);

    pbuf_free(p);

}

void Task_Eth50Hz_Init() {

	// Init UDP
	err_t retval;
	retval = udp_bind(&udpPcb50hz, IP4_ADDR_ANY, JETSON_50HZ_PORT);
	assert(retval == ERR_OK);
	udp_recv(&udpPcb50hz, Eth50Hz_Callback, NULL);
}

void Task_Eth50Hz(const Eth50HzInputMessage *inputMessage,
                   Eth50HzOutputMessage *outputMessage) {

	Eth50HzMessage *msgArray = outputMessage->OutMsgs;

	Eth50HzMessage ethRxMsg;
	size_t ethMsgCounter = 0;
	while (xQueueReceive(eth50HzRxQueue, &ethRxMsg, 0) == pdTRUE) {

		msgArray[ethMsgCounter] = ethRxMsg;
		++ethMsgCounter;

	}
	outputMessage->NumEscMsgs = ethMsgCounter;

}
