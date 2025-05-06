//
// Created by gleb on 4/12/25.
//

#include "messages.h"

void CAN100HzMessage_Init(CAN100HzMessage *m)
{
    m->NImuFrames = 0;
}

void Eth50HzMessage_Init(Eth50HzOutputMessage *m)
{
	for (size_t i = 0; i < CONF_N_ESCS; i++) {
		m->OutMsgs->EscEnable[i] = 0;
		m->OutMsgs->EscRpm[i] = 0;
	}
	m->NumEscMsgs = 0;
}
