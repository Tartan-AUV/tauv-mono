#include <Arduino.h>

#include "tx.h"

#define PIN_TX 2
#define PIN_TX_EN 3

static Config *config;

static TeensyTimerTool::PeriodicTimer carrier_timer;
static TeensyTimerTool::PeriodicTimer bit_timer;

static volatile bool transmitting_frame;
static volatile bool transmitting_buf;
static volatile bool transmitting_sync;
static uint8_t buf[FRAME_MAX_PAYLOAD_LENGTH + FRAME_META_LENGTH];
static size_t buf_length;

static volatile size_t sync_bit_index;
static volatile size_t buf_bit_index;

static void transmit_sync();
static void transmit_buf();
static void handle_carrier_timer();
static void handle_sync_bit_timer();
static void handle_buf_bit_timer();

void tx::setup(Config *new_config)
{
    config = new_config;

    transmitting_frame = false;
    transmitting_buf = false;
    transmitting_sync = false;

    memset(buf, 0, sizeof(buf));
    buf_length = 0;

    sync_bit_index = 0;
    buf_bit_index = 0;

    pinMode(PIN_TX, OUTPUT);
    pinMode(PIN_TX_EN, OUTPUT);
}

void tx::transmit(Frame *frame)
{
    if (transmitting_frame)
        return;

    transmitting_frame = true;

    frame->update_checksum();

    buf[0] = frame->sequence;
    memcpy(frame->payload, buf, frame->payload_length);
    buf[frame->payload_length + 1] = frame->checksum;

    buf_length = frame->payload_length + FRAME_META_LENGTH;
    buf_bit_index = 0;

    transmit_sync();

    transmit_buf();

    transmitting_frame = false;
}

static void transmit_sync()
{
    transmitting_sync = true;
    sync_bit_index = 0;

    digitalWriteFast(PIN_TX_EN, HIGH);
    digitalWriteFast(PIN_TX, LOW);

    carrier_timer.begin(handle_carrier_timer, config->half_period_hi_ns(), false);
    bit_timer.begin(handle_sync_bit_timer, config->period_sync_bit_ns());
    handle_sync_bit_timer();

    while (transmitting_sync)
        ;

    digitalWriteFast(PIN_TX_EN, LOW);
    digitalWriteFast(PIN_TX, LOW);
}

static void transmit_buf()
{
    transmitting_buf = true;
    buf_bit_index = 0;

    digitalWriteFast(PIN_TX_EN, HIGH);
    digitalWriteFast(PIN_TX, LOW);

    carrier_timer.begin(handle_carrier_timer, config->half_period_hi_ns(), false);
    bit_timer.begin(handle_buf_bit_timer, config->period_bit_ns());
    handle_buf_bit_timer();

    while (transmitting_buf)
    {
    }

    digitalWriteFast(PIN_TX_EN, LOW);
    digitalWriteFast(PIN_TX, LOW);
}

static void handle_carrier_timer()
{
    digitalWriteFast(PIN_TX, !digitalReadFast(PIN_TX));
}

static void handle_buf_bit_timer()
{
    if (buf_bit_index == 8 * buf_length)
    {
        carrier_timer.stop();
        bit_timer.stop();
        transmitting_buf = false;
        return;
    }

    size_t byte = buf[buf_bit_index / 8];
    bool bit = byte & (1 << (buf_bit_index % 8));

    std::chrono::nanoseconds period = bit ? config->half_period_hi_ns() : config->half_period_lo_ns();

    carrier_timer.setPeriod(period);

    if (buf_bit_index == 0)
    {
        carrier_timer.start();
    }

    ++buf_bit_index;
}

static void handle_sync_bit_timer()
{
    if (sync_bit_index == 0)
    {
        carrier_timer.setPeriod(config->half_period_hi_ns());
        carrier_timer.start();
    }
    else if (sync_bit_index == 1)
    {
        carrier_timer.setPeriod(config->half_period_lo_ns());
    }
    else
    {
        carrier_timer.stop();
        bit_timer.stop();
        transmitting_sync = false;
    }

    ++sync_bit_index;
}