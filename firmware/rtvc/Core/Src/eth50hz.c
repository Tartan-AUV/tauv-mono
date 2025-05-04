/* TAUV_FB TAUV */
/* 50Hz Ethernet (Receive) Task */
/* Author: Victor Zayakov */

#include "eth50hz.h"

#include "eth_msg_jetson_rtvc_builder.h"
#include "eth_msg_jetson_rtvc_reader.h"
#include "udp.h"
#include "vehicle_config.h"

static struct udp_pcb udpPcb50hz;

/* Global storage for the latest ESC/thruster command */
static float   g_thruster_rpms[MAX_THRUSTERS];
static uint8_t g_thruster_enabled[MAX_THRUSTERS];
static size_t  g_thruster_count = 0;

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
	if (!ThrusterCommand_verify_as_root(buf, len)) {
		pbuf_free(p);
		return;
	}

	/* Get root table */
	ThrusterCommand_table_t cmd = ThrusterCommand_as_root(buf);

	/* Pull out vectors, clamp to MAX_THRUSTERS */
	uoffset_t n = ThrusterCommand_rpm_length(cmd);
	if (n > MAX_THRUSTERS) n = MAX_THRUSTERS;

	for (uoffset_t i = 0; i < n; ++i) {
		g_thruster_rpms[i]    = ThrusterCommand_rpm_at(cmd, i);
		g_thruster_enabled[i] = ThrusterCommand_enabled_at(cmd, i);
	}
	g_thruster_count = n;

	pbuf_free(p);

}

void Task_Eth50Hz_Init() {

	// Init UDP
	err_t retval;
	retval = udp_bind(&udpPcb50hz, IP4_ADDR_ANY, 11004);
	assert(err == ERR_OK);
	udp_recv(udpPcb50hz, Eth50Hz_Callback, NULL);
}
