//
// Created by gleb on 4/12/25.
//

#include "messages.h"

void CAN100HzMessage_Init(CAN100HzMessage *m)
{
    m->NImuFrames = 0;
}

void Eth50HzMessage_Init(Eth50HzMessage *m)
{
	for (i = 0; i < CONF_N_ESCS; i++) {
		m->EscEnable[i] = 0;
		m->EscRpm[i] = 0;
	}
}
