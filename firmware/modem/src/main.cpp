//
// Created by Gleb Ryabtsev on 7/25/23.
//

#include "main.h"
#include "fsk_modulator.h"
#include "dsss_modulator.h"
#include "dsss_demodulator.h"
#include "packet_decoder.h"

#define RX_DBG

d_raw_t adc_raw_buf[RAW_BUF_SIZE];
d_sdft_t sdft_buf_1[SDFT_BUF_SIZE];
bool sdft_buf_1_ready = false;
d_sdft_t sdft_buf_2[SDFT_BUF_SIZE];
bool sdft_buf_2_ready = false;

uint8_t decoded_data[200];

TeensyTimerTool::PeriodicTimer fskTimer(TeensyTimerTool::GPT1);
TeensyTimerTool::PeriodicTimer demodTimer;
TeensyTimerTool::PeriodicTimer bitTimer(TeensyTimerTool::GPT2);

void demod_adc_it();

modem_config_t modemConfig{
    .freq_lo = 49000,
    .freq_hi = 54000, // 56k on rx, 60k on tx
    .chip_rate = 500,
};

FSKModulator mod{&modemConfig, &fskTimer, &bitTimer};
FSKDemodulator fsk_demod{&modemConfig, &demodTimer, RAW_BUF_SIZE, adc_raw_buf, sdft_buf_1, sdft_buf_2, demod_adc_it, SDFT_BUF_SIZE};
Barker7Sequence *code;
DSSSModulator *dsss_mod;
DSSSDemodulator *dsss_demod;
PacketDecoder *packetDecoder;

FSKModulator::m_word_t buf[] = "\0\0TARTANAUV\0";
//FSKModulator::m_word_t buf[] = {0b10101010};

static void fsk_dec_cplt1();
static void fsk_dec_cplt2();

void print_buf(int8_t *buf, size_t buf_size) {
    for (int i = 0; i < buf_size; i++) {
        Serial.printf("%d,", buf[i]);
    }
    Serial.println();
}

void setup() {
    Serial.begin(115000);
    Serial.println("Setup completed!");

    pinMode(PIN_DBG_1, OUTPUT);
    pinMode(PIN_DBG_2, OUTPUT);

    code = new Barker7Sequence(8);

    print_buf(code->samples, code->nsamples);
#ifdef RX_DBG
    dsss_demod = new DSSSDemodulator(&modemConfig, *code, 8);
    fsk_demod.init();
    fsk_demod.start(fsk_dec_cplt1, fsk_dec_cplt2);
    packetDecoder = new PacketDecoder(&modemConfig, 200);  // 200 bits!!!
#endif

#ifdef TX_DBG
    mod.setSigma(0.5);
    mod.init();
    dsss_mod = new DSSSModulator{&modemConfig, *code, &mod, 256};
    Serial.println("DSSS modulator created");
#endif
}

void loop() {
#ifdef TX_DBG
    Serial.println("Running");
    while(dsss_mod->busy()) {
//        Serial.println("Waiting");
//        delay(100);
    }
//    mod.transmit(buf, 1);
    delay(2000);
    dsss_mod->transmit(buf, 12);
//    Serial.println("Running");
#endif
#ifdef RX_DBG
    DSSSDemodulator::DemodStatus s;
    bool triggered = false;
   if (sdft_buf_1_ready) {
       sdft_buf_1_ready = false;
       dsss_demod->demodulate(sdft_buf_1, SDFT_BUF_SIZE, decoded_data, 200, true,
                              &s);
       triggered = true;
//       digitalWriteFast(PIN_DBG_2, dsss_demod->isLocked());
   } else if (sdft_buf_2_ready) {
       sdft_buf_2_ready = false;
       dsss_demod->demodulate(sdft_buf_2, SDFT_BUF_SIZE, decoded_data, 200, true,
                              &s);
       triggered = true;
//       digitalWriteFast(PIN_DBG_2, dsss_demod->isLocked());
   }

   if (triggered && s.read_length_bits > 0) {
       Serial.printf("Lock start: %d, lock end: %d, intermittent loss: %d\n",
                     s.lock_start, s.lock_end,
                     s.intermittent_lock_loss);
       packetDecoder->receive(decoded_data, s.read_length_bits,
                              s.lock_start == 0 && !s.intermittent_lock_loss);
       if(packetDecoder->available()) {
           uint8_t packet[32];
           packetDecoder->readPacket(packet, 32);
           packet[31] = '\0';
           Serial.printf("Packet received: %s\n", packet);
       }
//       print_buf((int8_t *) decoded_data, 64);
       memset(decoded_data, 0, 64);
   }

#endif
}



void demod_adc_it() {
    fsk_demod.handle_sample();
}

int counter = 0;
static void fsk_dec_cplt1() {
//    DBG_PRINT("FSK dec cplt 1\n");
//    if (counter++ == 3) {
//        print_buf((int8_t *) sdft_buf_1, SDFT_BUF_SIZE);
//        while(true)
//            ;
//    }
//    print_buf(code->samples, code->nsamples);
    sdft_buf_1_ready = true;
}

static void fsk_dec_cplt2() {
//    DBG_PRINT("FSK dec cplt 2\n");
//    print_buf((int8_t *) sdft_buf_2, SDFT_BUF_SIZE);
    sdft_buf_2_ready = true;
}