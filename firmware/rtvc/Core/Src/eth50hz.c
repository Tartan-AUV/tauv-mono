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

/* Global storage for the latest ESC/thruster command */
static float   g_thruster_rpms[CONF_N_ESCS];
static uint8_t g_thruster_enabled[CONF_N_ESCS];

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

    printf("\n");
    // 3) Pull out rpm and enabled arrays, clamp to MAX_THRUSTERS
    for (size_t i = 0; i < CONF_N_ESCS; ++i) {
        g_thruster_rpms   [i] = TAUV_FB_ThrusterCommand_rpm_get(cmd, i);
        g_thruster_enabled[i] = TAUV_FB_ThrusterCommand_enabled_get(cmd, i);
        printf(" enabled: %d", g_thruster_enabled[i]);
    }
    printf("\n");

    pbuf_free(p);

}

void Task_Eth50Hz_Init() {

	// Init UDP
	err_t retval;
	retval = udp_bind(&udpPcb50hz, IP4_ADDR_ANY, JETSON_50HZ_PORT);
	assert(retval == ERR_OK);
	udp_recv(&udpPcb50hz, Eth50Hz_Callback, NULL);
}
