/* TAUV TAUV */
/* 100Hz Ethernet Task */
/* Author: Gleb Ryabtsev */

#include "eth100hz.h"

#include "vehicle_config.h"
#include "rtvc_builder.h"
#include "udp.h"

static flatcc_builder_t    builder;
static struct udp_pcb      udpPcb100hz;

/* Task definition */

void Task_Eth100Hz_Init()
{
    flatcc_builder_init(&builder);

    // Init UDP
    err_t retval;
    retval = udp_bind(&udpPcb100hz, IP4_ADDR_ANY, 0);
    retval = udp_connect(&udpPcb100hz, &jetsonAddr, JETSON_100HZ_PORT);
    // todo: handle fault.
}

void Task_Eth100Hz(const Eth100HzInputMessage* inputMessage, Eth100HzMessage* outputMessage)
{
    // Payload construction
    const XsensIMUFrame *imuFrames = inputMessage->imuFrames;

    TAUV_Eth100HzMsg_start_as_root(&builder);
    TAUV_Eth100HzMsg_imu_data_start(&builder);
    for (size_t i = 0; i < inputMessage->nXsensImuFrames; ++i)
    {
        const XsensIMUFrame *imuFrame = &imuFrames[i];
        TAUV_XsensIMUFrame_t imuFrameBuf;

        imuFrameBuf.sample_time = imuFrames[i].SampleTime;
        copy_quat(imuFrame->Orientation, imuFrameBuf.orientation);
        copy_vec3(imuFrame->RateOfTurn, imuFrameBuf.rate_of_turn);
        copy_vec3(imuFrame->FreeAcceleration, imuFrameBuf.free_acceleration);

        TAUV_Eth100HzMsg_imu_data_push(&builder, &imuFrameBuf);
    }
    TAUV_Eth100HzMsg_imu_data_end(&builder);
    TAUV_Eth100HzMsg_end_as_root(&builder);
    size_t eth100HzMsgBufSize;
    const void* eth100HzMsgBuf = flatcc_builder_get_direct_buffer(&builder, &eth100HzMsgBufSize);
    assert(eth100HzMsgBuf);
    flatcc_builder_reset(&builder);

    // transmit the buffer
    struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, eth100HzMsgBufSize, PBUF_RAM);
    udp_sendto(&udpPcb100hz, p, &jetsonAddr, JETSON_100HZ_PORT);
    pbuf_free(p);
}
