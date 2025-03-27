#include "can100hz.h"
#include "main.h"
#include "vehicle_config.h"
#include "vesc.h"

void can_100hz_task(const CAN100HzInputMessage* input_msg, CAN100HzMessage* output_msg)
{
    const uint32_t rpm = 10000;
    HAL_StatusTypeDef status;
    /* ESC TX */
    for (size_t i = 0; i < 1; ++i)
    {
        const CAN_TxHeaderTypeDef esc_header = {
            .StdId = 0,
            .ExtId = vesc_get_can_msg_id(VESC_SET_RPM, (uint8_t) 79),
            .IDE = CAN_ID_EXT,
            .RTR = CAN_RTR_DATA,
            .DLC = 4,
            .TransmitGlobalTime = DISABLE,
        };

        uint8_t payload[4];
        vesc_get_rpm_payload(rpm, payload, sizeof(payload));

        uint32_t mailbox;

        HAL_CAN_AddTxMessage(&hcan1, &esc_header, payload, &mailbox);
    }

}