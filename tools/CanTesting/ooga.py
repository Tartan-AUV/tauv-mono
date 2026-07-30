#!/usr/bin/env python3
import can
import time
import struct
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

DTID_NODE_STATUS   = 341
DTID_GET_NODE_INFO = 1

HEALTH = {0: "OK", 1: "WARNING", 2: "ERROR", 3: "CRITICAL"}
MODE   = {0: "OPERATIONAL", 1: "INITIALIZATION", 2: "MAINTENANCE", 3: "SW_UPDATE", 7: "OFFLINE"}


def parse_can_id(can_id):
    service_flag = (can_id >> 7) & 1
    source_id    =  can_id & 0x7F
    if service_flag:
        return dict(
            service=True,
            dtid=(can_id >> 16) & 0xFF,
            is_request=bool((can_id >> 15) & 1),
            dest_id=(can_id >> 8) & 0x7F,
            source_id=source_id,
        )
    return dict(service=False, dtid=(can_id >> 8) & 0xFFFF, source_id=source_id)


def make_node_info_request(dest, our_id, tid):
    can_id = (30 << 24) | ((DTID_GET_NODE_INFO & 0xFF) << 16) | (1 << 15) | (dest << 8) | (1 << 7) | our_id
    return can.Message(arbitration_id=can_id, is_extended_id=True, data=bytes([0xC0 | (tid & 0x1F)]))


def decode_node_status(data):
    if len(data) < 7:
        return None
    try:
        uptime = struct.unpack_from("<I", data, 0)[0]
        b4     = data[4]
        vendor = struct.unpack_from("<H", data, 5)[0]
        return dict(uptime=uptime, health=(b4 >> 6) & 3, mode=(b4 >> 3) & 7, vendor=vendor)
    except Exception:
        return None


def decode_node_info(data):
    if len(data) < 24:
        return None
    try:
        o = 7
        sw_maj, sw_min, flags = data[o], data[o+1], data[o+2]
        o += 3
        vcs = struct.unpack_from("<I", data, o)[0]
        o += 4
        if flags & 1:
            o += 8
        hw_maj, hw_min = data[o], data[o+1]
        o += 2
        uid = " ".join("{:02X}".format(b) for b in data[o:o+16])
        o += 16
        coa_len = data[o]
        o += 1 + coa_len
        name = data[o:].decode("ascii", errors="replace").rstrip("\x00")
        return dict(
            sw="{}.{}".format(sw_maj, sw_min),
            hw="{}.{}".format(hw_maj, hw_min),
            vcs=vcs, uid=uid, name=name,
        )
    except Exception:
        return None


@dataclass
class Node:
    node_id: int
    health: str = "?"
    mode: str = "?"
    uptime: int = 0
    vendor: int = 0
    name: str = ""
    sw: str = ""
    hw: str = ""
    vcs: int = 0
    uid: str = ""
    first_seen: float = 0.0
    last_seen: float = 0.0
    _buf: dict = field(default_factory=dict)


class Scanner:
    def __init__(self, iface, duration, our_id, verbose):
        self.iface    = iface
        self.duration = duration
        self.our_id   = our_id
        self.verbose  = verbose
        self.nodes    = {}
        self._requested = set()
        self._tid = 0

    def _next_tid(self):
        t = self._tid & 0x1F
        self._tid += 1
        return t

    def _handle(self, msg, bus):
        if not msg.is_extended_id:
            return
        p   = parse_can_id(msg.arbitration_id)
        src = p["source_id"]
        now = time.monotonic()

        if not p["service"] and p["dtid"] == DTID_NODE_STATUS:
            ns = decode_node_status(bytes(msg.data[:-1]))
            if ns is None:
                return
            if src not in self.nodes:
                self.nodes[src] = Node(node_id=src, first_seen=now)
                h = HEALTH.get(ns["health"], "?")
                m = MODE.get(ns["mode"], "?")
                print("\r  [+] Node {:3d}  {:<16}  health: {}".format(src, m, h))
            rec = self.nodes[src]
            rec.health    = HEALTH.get(ns["health"], str(ns["health"]))
            rec.mode      = MODE.get(ns["mode"], str(ns["mode"]))
            rec.uptime    = ns["uptime"]
            rec.vendor    = ns["vendor"]
            rec.last_seen = now
            if src not in self._requested:
                self._requested.add(src)
                try:
                    bus.send(make_node_info_request(src, self.our_id, self._next_tid()))
                    if self.verbose:
                        print("\r  [>] GetNodeInfo sent to node {}".format(src))
                except can.CanError as e:
                    if self.verbose:
                        print("\r  [!] Send error: {}".format(e))

        elif p["service"] and not p["is_request"]:
            if p.get("dest_id") != self.our_id:
                return
            if p["dtid"] != DTID_GET_NODE_INFO:
                return
            if src not in self.nodes:
                return
            rec  = self.nodes[src]
            tail = msg.data[-1]
            pay  = bytes(msg.data[:-1])
            sot  = (tail >> 7) & 1
            eot  = (tail >> 6) & 1
            tid  = tail & 0x1F
            if sot:
                rec._buf[tid] = pay
            elif tid in rec._buf:
                rec._buf[tid] += pay
            if eot and tid in rec._buf:
                info = decode_node_info(rec._buf.pop(tid))
                if info:
                    rec.name = info["name"]
                    rec.sw   = info["sw"]
                    rec.hw   = info["hw"]
                    rec.vcs  = info["vcs"]
                    rec.uid  = info["uid"]
                    if self.verbose:
                        print("\r  [i] Node {}: {}  FW {}".format(src, rec.name, rec.sw))

    def run(self):
        print("")
        print("=" * 58)
        print("  DroneCAN Scanner -- " + self.iface)
        print("  Listening {}s  (AP_Periph beacons every ~1s)".format(int(self.duration)))
        print("=" * 58)
        print("")

        try:
            bus = can.interface.Bus(channel=self.iface, interface="socketcan")
        except Exception as e:
            print("[ERROR] Cannot open {}: {}".format(self.iface, e))
            print("")
            print("  Bring up the interface first:")
            print("    sudo ip link set {} up type can bitrate 1000000".format(self.iface))
            sys.exit(1)

        t_end   = time.monotonic() + self.duration
        spinner = list("/-\\|")
        i = 0
        try:
            while time.monotonic() < t_end:
                rem = t_end - time.monotonic()
                print("\r  {}  {:.1f}s left  |  {} node(s) found  ".format(
                    spinner[i % len(spinner)], rem, len(self.nodes)),
                    end="", flush=True)
                i += 1
                msg = bus.recv(timeout=0.1)
                if msg is not None:
                    self._handle(msg, bus)
        except KeyboardInterrupt:
            print("\n\n  Stopped.")
        finally:
            bus.shutdown()

        print("\r  Done.                                   ")
        print("")

    def print_report(self):
        nodes = sorted(self.nodes.values(), key=lambda r: r.node_id)
        print("=" * 58)
        print("  RESULTS -- {} node(s)".format(len(nodes)))
        print("=" * 58)
        print("")

        if not nodes:
            print("  No nodes detected.")
            print("")
            print("  Check:")
            print("  - Blue LED slow-blink = running  (fast = still booting)")
            print("  - 120 ohm terminator at BOTH ends of CAN bus")
            print("  - CAN-H to CAN-H, CAN-L to CAN-L")
            print("  - Raw traffic:  candump {}".format(self.iface))
            print("  - Wrong bitrate? Try 500k:")
            print("      sudo ip link set {} down".format(self.iface))
            print("      sudo ip link set {} up type can bitrate 500000".format(self.iface))
            return

        for rec in nodes:
            if rec.health == "OK":
                icon = "[OK]"
            elif rec.health == "WARNING":
                icon = "[WARN]"
            else[118;1:3u:
                icon = "[ERR]"
            print("  {}  Node {}".format(icon, rec.node_id))
            print("       Mode   : {}".format(rec.mode))
            print("       Health : {}".format(rec.health))
            print("       Uptime : {}s".format(rec.uptime))
            if rec.name:
                print("       Name   : {}".format(rec.name))
            if rec.sw:
                print("       FW     : {}  (0x{:08X})".format(rec.sw, rec.vcs))
            if rec.hw:
                print("       HW     : {}".format(rec.hw))
            if rec.uid:
                print("       UID    : {}".format(rec.uid))
            if rec.vendor:
                print("       Vendor : 0x{:04X}".format(rec.vendor))
            print("")


def main():
    p = argparse.ArgumentParser(description="DroneCAN scanner for AP_Periph CAN-L4-PWM")
    p.add_argument("--interface", "-i", default="can0")
    p.add_argument("--duration",  "-d", type=float, default=5.0)
    p.add_argument("--our-id",         type=int,   default=127)
    p.add_argument("--verbose",   "-v", action="store_true")
    args = p.parse_args()
    s = Scanner(args.interface, args.duration, args.our_id, args.verbose)
    s.run()
    s.print_report()


if __name__ == "__main__":
    main()
