/* TAUV RTVC */
/* 100Hz CAN Task */
/* Author: Gleb Ryabtsev */

#include "tasks.h"

#include "messages.h"
#include "can100hz.h"
#include "eth100hz.h"

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
}

void Task_1Hz() {

}

void Task_10Hz() {

}

// Persistent messages
static CAN100HzMessage can100HzMsg;

void Task_100Hz_Init()
{
    CAN100HzMessage_Init(&can100HzMsg);
}

void Task_100Hz() {

	Timestamp_t ts = get_timestamp();
	INFO("Secs: %d, usecs: %d\n", ts.secs, ts.usecs);

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
