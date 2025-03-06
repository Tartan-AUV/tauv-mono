/* TAUV RTVC */
/* 100Hz CAN Task */
/* Author: Gleb Ryabtsev */

#include "tasks.h"

void task1hz() {

}

void task10hz() {

}

void task100hz() {
    /* Input message buffers */
    CAN100HzInputMessage can_100hz_input_msg;
    Eth100HzInputMessage eth_100hz_input_msg;

    /* Output message buffers */
    CAN100HzMessage can_100hz_msg;
    Eth100HzMessage eth_100hz_msg;

    /* Execute tasks */
    RegularTaskStatus_t res;
    res = eth_100hz_task(&eth_100hz_input_msg, &eth_100hz_msg);

    can_100hz_input_msg = (CAN100HzInputMessage) {
        &eth_100hz_msg
    };

    res = can_100hz_task(&can_100hz_input_msg, &can_100hz_msg);
}

void task1000hz {

}