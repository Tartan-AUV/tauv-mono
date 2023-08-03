// #include <Arduino.h>

// void setup () {
//     Serial.begin(9600);
// }

// void loop () {
//     Serial.println("hello world\n");
// }


#define BLA
#ifdef BLA
#include "main.h"
#include "fsk_modulator.h"
#include "dsss_modulator.h"
#include "dsss_demodulator.h"
#include "packet_decoder.h"

#define MODE_TX


d_raw_t adc_raw_buf[RAW_BUF_SIZE];
d_sdft_t sdft_buf_1[SDFT_OUT_BUF_SIZE];
bool sdft_buf_1_ready = false;
d_sdft_t sdft_buf_2[SDFT_OUT_BUF_SIZE];
bool sdft_buf_2_ready = false;

uint8_t rx_buf_bits[RX_BUF_SIZE * 8];
uint8_t hdr[] = {0x00, 0xFF};

TeensyTimerTool::PeriodicTimer fskTimer(TeensyTimerTool::GPT1);
// TeensyTimerTool::PeriodicTimer demodTimer;
TeensyTimerTool::PeriodicTimer bitTimer(TeensyTimerTool::GPT2);

void demod_adc_it();

modem_config_t modemConfig{
        .freq_lo = 49000,
        .freq_hi = 54000,
        .chip_rate = CHIP_RATE,
};

FSKModulator mod{&modemConfig, &fskTimer, &bitTimer};
// FSKDemodulator fsk_demod{&modemConfig, &demodTimer, RAW_BUF_SIZE, adc_raw_buf, sdft_buf_1, sdft_buf_2, demod_adc_it,
//                          SDFT_OUT_BUF_SIZE};
Barker7Sequence *code;
DSSSModulator *dsss_mod;
// DSSSDemodulator *dsss_demod;
// PacketDecoder *packetDecoder;

static void fsk_dec_cplt1();

static void fsk_dec_cplt2();

void print_buf(int8_t *buf, size_t buf_size) {
    for (int i = 0; i < buf_size; i++) {
        Serial.printf("%d,", buf[i]);
    }
    Serial.println();
}

void setup() {
    Serial.begin(9600);
    delay(1000);
    Serial.println("Ready");

    pinMode(PIN_DBG_1, OUTPUT);
    pinMode(PIN_DBG_2, OUTPUT);

    code = new Barker7Sequence(8);

#ifdef MODE_RX
    dsss_demod = new DSSSDemodulator(&modemConfig, *code, 8);
    fsk_demod.init();
    fsk_demod.start(fsk_dec_cplt1, fsk_dec_cplt2);
#endif

#ifdef MODE_TX
    mod.setSigma(0.5);
    mod.init();
    dsss_mod = new DSSSModulator{&modemConfig, *code, &mod, 256};
#endif
}

void loop() {
#ifdef MODE_TX
    if (Serial.available()) {
        uint8_t serial_in_buf[PACKET_SIZE];
        memset(serial_in_buf, 0, PACKET_SIZE);
        Serial.readBytesUntil('\n', serial_in_buf, PACKET_SIZE);
        // for (int i = 0; i < PACKET_SIZE; i++) {
        //     Serial.print(serial_in_buf[i]);
        // }
        // Serial.println();
        // Serial.println("Transmitting.");
        uint8_t tx_out_buf[TX_BUF_SIZE];
        memcpy(tx_out_buf + 2, serial_in_buf, PACKET_SIZE);
        // Serial.println("Waiting");
        while (dsss_mod->busy()) 
            delay(10);
        // Serial.println("Transmitting...");
        dsss_mod->transmit(tx_out_buf, TX_BUF_SIZE);
    } 
    delay (10);
#endif

#ifdef MODE_RX
    DSSSDemodulator::DemodStatus s;
    bool triggered = false;
    if (sdft_buf_1_ready) {
        sdft_buf_1_ready = false;
        memset(rx_buf_bits, 0, RX_BUF_SIZE * 8);
        dsss_demod->demodulate(sdft_buf_1, SDFT_OUT_BUF_SIZE, rx_buf_bits, RX_BUF_SIZE * 8, true,
                                &s);
        triggered = true;
    } else if (sdft_buf_2_ready) {
        sdft_buf_2_ready = false;
        memset(rx_buf_bits, 0, RX_BUF_SIZE * 8);
        dsss_demod->demodulate(sdft_buf_2, SDFT_OUT_BUF_SIZE, rx_buf_bits, RX_BUF_SIZE * 8, true,
                                &s);
        triggered = true;
    }
    if (triggered && s.read_length_bits > 0) {
//            Serial.printf("Lock start: %d, lock end: %d, intermittent loss: %d\n",
//                          s.lock_start, s.lock_end,
//                          s.intermittent_lock_loss);
    }
#endif

}

void demod_adc_it() {
    // fsk_demod.handle_sample();
}

int counter = 0;

static void fsk_dec_cplt1() {
//    DBG_PRINT("FSK dec cplt 1\n");
//    if (counter++ == 3) {
//        print_buf((int8_t *) sdft_buf_1, SDFT_OUT_BUF_SIZE);
//        while(true)
//            ;
//    }
//    print_buf(code->samples, code->nsamples);
    sdft_buf_1_ready = true;
}

static void fsk_dec_cplt2() {
//    DBG_PRINT("FSK dec cplt 2\n");
//    print_buf((int8_t *) sdft_buf_2, SDFT_OUT_BUF_SIZE);
    sdft_buf_2_ready = true;
}

#endif