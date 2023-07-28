//
// Created by Gleb Ryabtsev on 7/28/23.
//

#include "packet_decoder.h"

PacketDecoder::PacketDecoder(modem_config_t *modemConfig, size_t max_packet_buf_size) :
        modemConfig(modemConfig), packet_buf_size(max_packet_buf_size) {
    packet_buf = new uint8_t[max_packet_buf_size];
    context_bits = new uint8_t[header_bits_length];
    packet_size = 0;
    bits_received = 0;
}

PacketDecoder::ReceiveStatus PacketDecoder::receive(uint8_t *bitbuf, size_t size, bool continous) {
    if (bits_received == 0 && packet_size > 0) {
        return ERROR;
        // packet waiting to be read
    }

    memset(packet_buf, 0, packet_buf_size);

    ReceiveStatus status;
    int32_t shift;
    if (bits_received == 0 || continous) {
        DBG_PRINT("Looking for header\n");
        int32_t header_pos = find_header(bitbuf, size);
        if (header_pos == -1) {
            status = NO_PACKET;
            DBG_PRINT("No header found\n");
            DBG_PRINT("Attempting memcpy, size: %d\n", size);
            memcpy(context_bits, bitbuf + size - header_bits_length, header_bits_length);
            return status;
        }
        DBG_PRINT("Header found at %d\n", header_pos);  ;
        status = IN_PROGRESS;
        shift = header_pos + header_bits_length;
        packet_size = 0;
    } else {
        DBG_PRINT("Continuing packet\n");
        shift = -bits_received;
    }

    while ((bits_received + shift < size) &&
            (bits_received < packet_size || packet_size == 0)) {
        packet_buf[bits_received / 8] |= (bits_received % 8);
        bits_received++;
        if (bits_received == 8) {
            DBG_PRINT("Received size: %d\n", packet_buf[0]);
            packet_size = packet_buf[0];
        }
    }

    DBG_PRINT("Bits received: %d\n", bits_received);

    if (bits_received == packet_size) {
        status = DONE;
        bits_received = 0;
    } else {
        status = IN_PROGRESS;
    }
    memcpy(context_bits, bitbuf + size - header_bits_length, header_bits_length);
    return status;
}

int32_t PacketDecoder::find_header(uint8_t *bitbuf, size_t size) {
    for (int i = -header_bits_length+1; i < size - header_bits_length; i++) {
        bool found = true;
        for (int j = 0; j < header_bits_length; j++) {
            if (get_bit(i+j) != header_bits[j]) {
                found = false;
                break;
            }
        }
        if (found) {
            return i;
        }
    }
    return -1;
}

uint8_t PacketDecoder::get_bit(int32_t i) {
    if (i < 0) {
        return context_bits[header_bits_length + i];
    } else {
        return context_bits[i];
    }
}

uint8_t PacketDecoder::make_byte(uint8_t *bits) {
    uint8_t byte = 0;
    for (int i = 0; i < 8; i++) {
        byte |= bits[i] << i; // little-endian
    }
    return byte;
}