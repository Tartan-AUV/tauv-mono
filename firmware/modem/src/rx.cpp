#include <Arduino.h>
#include <ADC.h>
#include <math.h>
#include <complex.h>

#include "rx.h"

#define PIN_RX 14
#define PIN_RX_CLK 4
#define PIN_RX_DEMOD_DATA 5
#define PIN_RX_DEMOD_DATA_RAW 6
#define PIN_RX_DEMOD_BIT_CLOCK 7

static TeensyTimerTool::PeriodicTimer sample_timer(TeensyTimerTool::PIT);
static TeensyTimerTool::OneShotTimer timeout_timer(TeensyTimerTool::PIT);

static ADC adc;
static Config *config;

static volatile bool listening;

static void handle_sample_timer();
static void handle_timeout_timer();
static void handle_sample_ready();

static void reset();

static volatile float *sample_buf;
static size_t sample_buf_length;
static volatile size_t sample_buf_index;

static size_t N;
static float r;

static float k_lo;
static float k_hi;

static float coeff_w[3];

static float coeff_a;
static __complex__ float coeff_b_lo[3];
static __complex__ float coeff_b_hi[3];

static volatile __complex__ float s_lo_w[3];
static volatile __complex__ float s_hi_w[3];

static volatile __complex__ float s_lo;
static volatile __complex__ float s_hi;

static volatile float mag_lo;
static volatile float mag_hi;

static volatile bool receiving_sync_hi;
static volatile bool receiving_sync_lo;
static volatile bool receiving;
static volatile bool current_bit;

static volatile uint32_t current_bit_time;
static volatile uint32_t sync_time;
static volatile uint32_t next_bit_time;
static volatile uint32_t bit_count;
static volatile uint32_t bit_sample_count;

static uint8_t rx_buf[FRAME_MAX_PAYLOAD_LENGTH + FRAME_META_LENGTH];
static size_t rx_bit_index;
static size_t rx_buf_size = 0;

static volatile float slow_bit_avg;
static volatile float fast_bit_avg;

void rx::setup(Config *new_config)
{
    config = new_config;
    adc.adc0->setAveraging(0);
    adc.adc0->setResolution(8);
    adc.adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
    adc.adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);
    adc.adc0->enableInterrupts(handle_sample_ready);

    sample_buf_length = config->sdft_N;
    sample_buf = (float *)malloc(sizeof(float) * sample_buf_length);
    memset((void *)sample_buf, 0, sizeof(float) * sizeof(sample_buf_length));

    N = config->sdft_N;
    r = (float)config->sdft_r;

    k_lo = ((float)config->freq_lo * (float)N) / ((float)config->freq_sample);
    k_hi = ((float)config->freq_hi * (float)N) / ((float)config->freq_sample);

    coeff_w[0] = -0.25;
    coeff_w[1] = 0.5;
    coeff_w[2] = -0.25;

    coeff_a = -pow(r, (float)N);

    coeff_b_lo[0] = r * cexp((2.i * PI * (k_lo - 1.)) / (float)N);
    coeff_b_lo[1] = r * cexp((2.i * PI * k_lo) / (float)N);
    coeff_b_lo[2] = r * cexp((2.i * PI * (k_lo + 1.)) / (float)N);
    coeff_b_hi[0] = r * cexp((2.i * PI * (k_hi - 1.)) / (float)N);
    coeff_b_hi[1] = r * cexp((2.i * PI * k_hi) / (float)N);
    coeff_b_hi[2] = r * cexp((2.i * PI * (k_hi + 1.)) / (float)N);

    pinMode(PIN_RX_CLK, OUTPUT);
    pinMode(PIN_RX_DEMOD_DATA, OUTPUT);
    pinMode(PIN_RX_DEMOD_DATA_RAW, OUTPUT);
    pinMode(PIN_RX_DEMOD_BIT_CLOCK, OUTPUT);
}

bool rx::receive(Frame *frame, std::chrono::nanoseconds timeout)
{
    listening = true;

    reset();

    timeout_timer.begin(handle_timeout_timer);

    timeout_timer.trigger(timeout);

    sample_timer.begin(handle_sample_timer, config->period_sample_ns());

    while (listening)
        ;

    if (receiving)
    {
        frame->payload_length = rx_buf[0];
        memcpy(frame->payload, &rx_buf[1], frame->payload_length);
        frame->checksum = rx_buf[frame->payload_length + 1];
    }

    return receiving;
}

static void handle_sample_timer()
{
    adc.adc0->startSingleRead(PIN_RX);

    digitalWriteFast(PIN_RX_CLK, HIGH);
}

static void handle_timeout_timer()
{
    Serial.println("timeout");
    sample_timer.stop();
    listening = false;
}

static void reset()
{
    sample_buf_index = 0;
    rx_bit_index = 0;
    receiving = 0;
    receiving_sync_hi = 0;
    receiving_sync_lo = 0;

    fast_bit_avg = 0;
    slow_bit_avg = 0;

    bit_count = 0;
    bit_sample_count = 0;

    uint32_t time = micros() * (uint32_t)ONE_US_IN_NS.count();

    current_bit_time = time;
    sync_time = time;

    memset((void *)rx_buf, 0, sizeof(rx_buf));
    rx_buf_size = 0;

    memset((void *)sample_buf, 0, sizeof(float) * sample_buf_length);
    memset((void *)s_lo_w, 0, sizeof(s_lo_w));
    memset((void *)s_hi_w, 0, sizeof(s_lo_w));
}

static void handle_sample_ready()
{
    uint32_t time = micros() * (uint32_t)ONE_US_IN_NS.count();

    uint8_t sample_raw = adc.adc0->readSingle();

    digitalWriteFast(PIN_RX_CLK, LOW);

    float sample = ((float)sample_raw) / UINT8_MAX;

    sample_buf[sample_buf_index] = sample;

    float sample_N = sample_buf[(sample_buf_index + 1) % sample_buf_length];

    float a = sample - coeff_a * sample_N;

    s_lo_w[0] = a + coeff_b_lo[0] * s_lo_w[0];
    s_lo_w[1] = a + coeff_b_lo[1] * s_lo_w[1];
    s_lo_w[2] = a + coeff_b_lo[2] * s_lo_w[2];

    s_hi_w[0] = a + coeff_b_hi[0] * s_hi_w[0];
    s_hi_w[1] = a + coeff_b_hi[1] * s_hi_w[1];
    s_hi_w[2] = a + coeff_b_hi[2] * s_hi_w[2];

    s_lo = coeff_w[0] * s_lo_w[0] + coeff_w[1] * s_lo_w[1] + coeff_w[2] * s_lo_w[2];
    s_hi = coeff_w[0] * s_hi_w[0] + coeff_w[1] * s_hi_w[1] + coeff_w[2] * s_hi_w[2];

    mag_lo = cabs(s_lo);
    mag_hi = cabs(s_hi);

    sample_buf_index = (sample_buf_index + 1) % sample_buf_length;

    bool raw_bit = mag_hi > mag_lo;

    slow_bit_avg = config->slow_bit_coeff * (float)(raw_bit ? 1.0 : 0.0) + (1 - config->slow_bit_coeff) * slow_bit_avg;
    fast_bit_avg = config->fast_bit_coeff * (float)(raw_bit ? 1.0 : 0.0) + (1 - config->fast_bit_coeff) * fast_bit_avg;

    bool bit = slow_bit_avg > 0.5;

    if (bit != current_bit)
    {
        current_bit = bit;
        current_bit_time = time;
    }

    if (!receiving)
    {
        if (!receiving_sync_hi && !receiving_sync_lo && current_bit && (time - current_bit_time) > (uint32_t)config->period_sync_bit_min_ns().count())
        {
            // Caught sync high
            receiving_sync_hi = true;

            Serial.println("syncing");
        }
        else if (receiving_sync_hi && !receiving_sync_lo && (fast_bit_avg - slow_bit_avg) < config->edge_threshold)
        {
            Serial.println("synced");
            // Caught sync falling edge
            sync_time = time;
            receiving_sync_hi = false;
            receiving_sync_lo = true;
        }
        else if (!receiving_sync_hi && receiving_sync_lo && time > (sync_time + (uint32_t)config->period_sync_bit_ns().count()))
        {
            // Finished sync low
            receiving = true;
            receiving_sync_hi = false;
            receiving_sync_lo = false;

            next_bit_time = time + (uint32_t)config->period_bit_ns().count();

            Serial.println("start receiving");
        }
    }
    else
    {
        if (time > next_bit_time)
        {
            digitalWriteFast(PIN_RX_DEMOD_BIT_CLOCK, !digitalReadFast(PIN_RX_DEMOD_BIT_CLOCK));

            bool out_bit = bit_count > (bit_sample_count / 2);
            // bool out_bit = current_bit;

            rx_buf[rx_bit_index / 8] |= (out_bit << (rx_bit_index % 8));
            rx_bit_index++;

            if (rx_bit_index >= 8 && rx_bit_index >= 8 * (rx_buf[0] + 1))
            {
                listening = false;
            }

            if (rx_bit_index % 8 == 0)
            {
                rx_buf_size++;
            }

            next_bit_time = next_bit_time + (uint32_t)config->period_bit_ns().count();
            bit_count = 0;
            bit_sample_count = 0;
        }

        bit_sample_count++;

        if (current_bit)
            bit_count++;
    }

    digitalWriteFast(PIN_RX_DEMOD_DATA, bit);
    digitalWriteFast(PIN_RX_DEMOD_DATA_RAW, raw_bit);
}