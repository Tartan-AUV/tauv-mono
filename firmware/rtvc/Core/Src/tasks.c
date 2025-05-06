/* TAUV RTVC */
/* 100Hz CAN Task */
/* Author: Gleb Ryabtsev */

#include "tasks.h"

#include "messages.h"
#include "can100hz.h"
#include "eth100hz.h"
#include "eth50hz.h"
#include "can50hz.h"

#include "timekeeping.h"
#include "logging.h"

/* PFP */
void Task_100Hz_Init();

void TasksInit()
{
    CAN_RxQueueInit();

    Task_100Hz_Init();
    Task_CAN100Hz_Init();
    Task_Eth100Hz_Init();

    // 50 Hz ethernet task (receives from Jetson)
    Task_Eth50Hz_Init();
    // Top-level 50 Hz task
    Task_50Hz_Init();
}

void Task_1Hz() {

}

void Task_10Hz() {

}

// Persistent Ethernet message
static Eth50HzOutputMessage eth50HzOutputMsg;

void Task_50Hz_Init()
{
	Eth50HzMessage_Init(&eth50HzOutputMsg);
}

void Task_50Hz() {

	const CAN50HzInputMessage can50HzInputMsg = {
		.outputMsg = eth50HzOutputMsg,
	};
	CAN50HzMessage can50HzMsg;

	Task_CAN50Hz(&can50HzInputMsg, &can50HzMsg);

	const Eth50HzInputMessage eth50HzInputMsg = {
		.placeholder = 0,
	};
	Task_Eth50Hz(&eth50HzInputMsg, &eth50HzOutputMsg);

}

// Persistent messages
static CAN100HzMessage can100HzMsg;

void Task_100Hz_Init()
{
    CAN100HzMessage_Init(&can100HzMsg);
}

void Task_100Hz() {

	Timestamp_t ts = get_timestamp();

    const Eth100HzInputMessage eth100HzInputMsg = {
        .imuFrames = can100HzMsg.ImuFrames,
        .nXsensImuFrames = can100HzMsg.NImuFrames,
    };
    Eth100HzMessage eth100HzMsg;

    Task_Eth100Hz(&eth100HzInputMsg, &eth100HzMsg);

    const CAN100HzInputMessage can100HzInputMsg = {
        .eth_100hz_msg = &eth100HzMsg,
    };

    Task_CAN100Hz(&can100HzInputMsg, &can100HzMsg);

}

void Task_1000Hz() {

}
