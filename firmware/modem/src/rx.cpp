#include <Arduino.h>
#include <ADC.h>
#include <math.h>
#include <complex.h>

#include "rx.h"

#define PIN_RX 14
#define PIN_RX_CLK 4
#define PIN_RX_DEMOD_DATA 5
#define PIN_RX_DEMOD_VALID 6

static TeensyTimerTool::PeriodicTimer sample_timer(TeensyTimerTool::PIT);
static TeensyTimerTool::OneShotTimer timeout_timer;

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

static volatile bool receiving_sync;
static volatile bool receiving;
static volatile bool current_bit;
static volatile uint current_bit_duration;

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
    pinMode(PIN_RX_DEMOD_VALID, OUTPUT);
}

void rx::receive(Frame *frame, std::chrono::nanoseconds timeout)
{
    listening = true;

    reset();

    timeout_timer.begin(handle_timeout_timer);

    timeout_timer.trigger(timeout);

    sample_timer.begin(handle_sample_timer, config->period_sample_ns());

    while (listening)
        ;

    Serial.printf("%f %f\n", mag_lo, mag_hi);
}

static void handle_sample_timer()
{
    adc.adc0->startSingleRead(PIN_RX);

    digitalWriteFast(PIN_RX_CLK, HIGH);
}

static void handle_timeout_timer()
{
    sample_timer.stop();
    listening = false;
}

static void reset()
{
    sample_buf_index = 0;

    memset((void *)sample_buf, 0, sizeof(float) * sizeof(sample_buf_length));
    memset((void *)s_lo_w, 0, sizeof(s_lo_w));
    memset((void *)s_hi_w, 0, sizeof(s_lo_w));
}

static void handle_sample_ready()
{
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

    // TODO: Add some sort of hysterisis here
    bool bit = mag_hi > mag_lo;

    if (bit == current_bit)
    {
        ++current_bit_duration;
    }
    else
    {
        current_bit = bit;
        current_bit_duration = 0;
    }

    if (!receiving && !receiving_sync && bit && current_bit_duration > config->sync_bit_duration)
    {
    }

    // After high for sync bit duration, set receiving sync
    // On falling edge, set receiving sync off and set receiving
    // Then take majority over each bit window for next 8 bits
    // Repeat to get multiple bytes of received data (expect sync pulses between bytes)

    digitalWriteFast(PIN_RX_DEMOD_DATA, bit);
}