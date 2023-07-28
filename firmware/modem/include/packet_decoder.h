//
// Created by Gleb Ryabtsev on 7/28/23.
//

#ifndef MODEM_PACKET_DECODER_H
#define MODEM_PACKET_DECODER_H

#include <main.h>

class PacketDecoder {
public:
    enum ReceiveStatus {
        NO_PACKET,
        IN_PROGRESS,
        DONE,
        ERROR
    };

    PacketDecoder(modem_config_t *modemConfig, size_t max_packet_buf_size);

    PacketDecoder::ReceiveStatus receive(uint8_t *bitbuf, size_t size, bool continous);

    bool available() {
        return bits_received == 0 && packet_size > 0;
    }

    size_t readPacket(uint8_t *buf, size_t size) {
        size_t read_size = packet_size < size ? packet_size : size;
        memcpy(buf, packet_buf, read_size);
        packet_size = 0;
        return read_size;
    }

private:
    uint8_t *packet_buf;
    size_t packet_buf_size;

    size_t packet_size;
    size_t bits_received;

    uint8_t header_bits[24] = {0, 1, 0, 0, 0, 1, 1, 0,
                               0, 1, 0, 0, 1, 1, 1, 0,
                               0, 1, 0, 1, 0, 0, 0, 1};
    uint8_t header_bits_length = 24;

    uint8_t *context_bits;

    modem_config_t *modemConfig;

    int32_t find_header(uint8_t *bitbuf, size_t size);

    uint8_t get_bit(int32_t i);

    uint8_t make_byte(uint8_t *bits);
};



#endif //MODEM_PACKET_DECODER_H
