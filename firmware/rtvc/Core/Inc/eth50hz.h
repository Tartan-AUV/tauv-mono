/* TAUV RTVC */
/* 50Hz Ethernet (Receive) Task */
/* Author: Victor Zayakov */

/* INCLUDES */
#ifndef ETH50HZ_H
#define ETH50HZ_H

#include "messages.h"


/* TASK DECLARATION */
void Task_Eth50Hz_Init();

void Eth50Hz_Callback(void *arg,
		  struct udp_pcb *pcb,
		  struct pbuf *p,
		  const ip_addr_t *addr,
		  u16_t port);

//void Task_Eth50Hz(Eth50HzMessage* outputMessage);

#endif // ETH50HZ_H
