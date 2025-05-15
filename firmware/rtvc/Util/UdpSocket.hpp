/******************************************************************************
 *  TartanAUV - Carnegie Mellon University
 *  RTVC Firmware
 *
 *  Author:      gleb
 *  Date:        5/14/25
 *
 *  Description:
 *      TODO
 *
 *****************************************************************************/
 
#pragma once

#include <functional>
#include <cstdint>
#include <cstring>
#include <lwip/udp.h>
#include <lwip/ip.h>
#include <lwip/pbuf.h>
#include <lwip/mem.h>
#include <lwip/inet.h>

using std::size_t;

class UdpSocket {
public:
    using ReceiveCallback = std::function<void(const ip_addr_t& from_addr, uint16_t from_port, const uint8_t* data, uint16_t len)>;

    void init() {
      pcb_ = udp_new();
      if (!pcb_) {
        Error_Handler();
      }
    }

    ~UdpSocket() {
        if (pcb_) {
            udp_remove(pcb_);
            pcb_ = nullptr;
        }
    }

    void bind(uint16_t port) {
        if (udp_bind(pcb_, IP_ADDR_ANY, port) != ERR_OK) {
          Error_Handler();
        }
    }

    void set_receive_callback(ReceiveCallback cb) {
        rx_callback_ = std::move(cb);
        udp_recv(pcb_, &UdpSocket::udp_rx_trampoline, this);
    }

    void send(const ip_addr_t& dest_ip, uint16_t dest_port, const uint8_t* data, uint16_t len) {
        pbuf* p = pbuf_alloc(PBUF_TRANSPORT, len, PBUF_RAM);
        if (!p) {
          Error_Handler();
        }

        std::memcpy(p->payload, data, len);
        udp_sendto(pcb_, p, &dest_ip, dest_port);
        pbuf_free(p);
    }

    void send(const char* ip_str, uint16_t port, const uint8_t* data, uint16_t len) {
        ip_addr_t addr;
        if (!ipaddr_aton(ip_str, &addr)) {
          Error_Handler();
        }
        send(addr, port, data, len);
    }

private:
    static void udp_rx_trampoline(void* arg, udp_pcb* /*pcb*/, pbuf* p,
                                  const ip_addr_t* addr, u16_t port) {
        auto* self = static_cast<UdpSocket*>(arg);
        if (self->rx_callback_ && p) {
            self->rx_callback_(*addr, port, static_cast<uint8_t*>(p->payload), p->len);
        }
        pbuf_free(p);
    }

    udp_pcb* pcb_ = nullptr;
    ReceiveCallback rx_callback_;
};
