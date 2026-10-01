#!/usr/bin/env python3
"""
DroneCAN Node Scanner — raw SocketCAN, no spin() deadlock.
Listens for NodeStatus broadcasts from AP_Periph / DroneCAN nodes
and sends GetNodeInfo requests to retrieve firmware details.

Usage:
    python3 dronecan_scan.py [--interface can0] [--duration 5]

Requires:
    pip install python-can
    sudo ip link set can0 up type can bitrate 1000000
"""

import can
import time
import struct
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# DroneCAN / UAVCAN v0 constants
# ---------------------------------------------------------------------------

DTID_NODE_STATUS   = 341
DTID_GET_NODE_INFO = 1

HEALTH = {0: "OK", 1: "WARNING", 2: "ERROR", 3: "CRITICAL"}
MODE   = {0: "OPERATIONAL", 1: "INITIALIZATION", 2: "MAINTENANCE",
          3: "SW_UPDATE", 7: "OFFLINE"}

# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def parse_can_id(can_id: int) -> dict:
    service_flag = (can_id >> 7) & 1
    source_id    =  can_id       & 0x7F
    if service_flag:
        return dict(service=True,
                    dtid     = (can_id >> 16) & 0xFF,
                    is_request = bool((can_id >> 15) & 1),
                    dest_id  = (can_id >>  8) & 0x7F,
                    source_id= source_id)
    else:
        return dict(service=False,
                    dtid     = (can_id >> 8) & 0xFFFF,
                    source_id= source_id)


def make_node_info_request(dest: int, our_id: int, tid: int) -> can.Message:
    can_id = ((30 << 24) | ((DTID_GET_NODE_INFO & 0xFF) << 16) |
              (1 << 15) | (dest << 8) | (1 << 7) | our_id)
    return can.Message(arbitration_id=can_id, is_extended_id=True,
                       data=bytes([0xC0 | (tid & 0x1F)]))


def decode_node_status(data: bytes) -> Optional[dict]:
    if len(data) < 7:
        return None
    try:
        uptime = struct.unpack_from("<I", data, 0)[0]
        b4     = data[4]
        vendor = struct.unpack_from("<H", data, 5)[0]
        return dict(uptime=uptime,
                    health=(b4 >> 6) & 3,
                    mode  =(b4 >> 3) & 7,
                    vendor=vendor)
    except Exception:
        return None


def decode_node_info(data: bytes) -> Optional[dict]:
    if len(data) < 24:
        return None
    try:
        o = 7
        sw_maj, sw_min, flags = data[o], data[o+1], data[o+2]; o += 3
        vcs = struct.unpack_from("<I", data, o)[0]; o += 4
        if flags & 1: o += 8
        hw_maj, hw_min = data[o], data[o+1]; o += 2
        uid = " ".join(f"{b:02X}" for b in data[o:o+16]); o += 16
        coa_len = data[o]; o += 1 + coa_len
        name = data[o:].decode("ascii", errors="replace").rstrip("\x00")
        return dict(sw=f"{sw_maj}.{sw_min}", hw=f"{hw_maj}.{hw_min}",
                    vcs=vcs, uid=uid, name=name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Node record
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class Scanner:
    def __init__(self, iface, duration, our_id, verbose):
        self.iface    = iface
        self.duration = duration
        self.our_id   = our_id
        self.verbose  = verbose
        self.nodes: dict[int, Node] = {}
        self._requested: set[int] = set()
        self._tid = 0

    def _next_tid(self):
        t = self._tid & 0x1F
        self._tid += 1
        return t

    def _handle(self, msg: can.Message, bus: can.Bus):
        if not msg.is_extended_id:
            return
        p   = parse_can_id(msg.arbitration_id)
        src = p["source_id"]
        now = time.monotonic()

        # -- NodeStatus broadcast --
        if not p["service"] and p["dtid"] == DTID_NODE_STATUS:
            ns = decode_node_status(bytes(msg.data[:-1]))
            if ns is None:
                return
            if src not in self.nodes:
                self.nodes[src] = Node(node_id=src, first_seen=now)
                print(f"\r  [+] Node {src:3d}  "
                      f"{MODE.get(ns['mode'],'?'):<16}  "
                      f"health: {HEALTH.get(ns['health'],'?')}")
            rec = self.nodes[src]
            rec.health  = HEALTH.get(ns["health"], str(ns["health"]))
            rec.mode    = MODE.get(ns["mode"], str(ns["mode"]))
            rec.uptime  = ns["uptime"]
            rec.vendor  = ns["vendor"]
            rec.last_seen = now
            if src not in self._requested:
                self._requested.add(src)
                try:
                    bus.send(make_node_info_request(src, self.our_id, self._next_tid()))
                    if self.verbose:
                        print(f"\r  [>] GetNodeInfo sent → node {src}")
                except can.CanError as e:
                    if self.verbose:
                        print(f"\r  [!] Send error: {e}")

        # -- Service response --
        elif p["service"] and not p["is_request"]:
            if p.get("dest_id") != self.our_id: return
            if p["dtid"] != DTID_GET_NODE_INFO:  return
            if src not in self.nodes:             return
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
                        print(f"\r  [i] Node {src}: {rec.name}  FW {rec.sw}")

    def run(self):
        print(f"\n{'='*58}")
        print(f"  DroneCAN Scanner — {self.iface}")
        print(f"  Listening {self.duration:.0f}s (AP_Periph beacons every ~1s)")
        print(f"{'='*58}\n")

        try:
            bus = can.interface.Bus(channel=self.iface, interface="socketcan")
        except Exception as e:
[118;1:3u            print(f"[ERROR] Cannot open {self.iface}: {e}")
            print(f"\n  Run first:")
            print(f"    sudo ip link set {self.iface} up type can bitrate 1000000")
            sys.exit(1)

        t_end   = time.monotonic() + self.duration
        spinner = list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        i = 0
        try:
            while time.monotonic() < t_end:
                rem = t_end - time.monotonic()
                print(f"\r  {spinner[i%len(spinner)]}  "
                      f"{rem:4.1f}s left  |  {len(self.nodes)} node(s)  ",
                      end="", flush=True)
                i += 1
                msg = bus.recv(timeout=0.1)   # non-blocking, returns None on timeout
                if msg is not None:
                    self._handle(msg, bus)
        except KeyboardInterrupt:
            print("\n\n  Stopped.")
        finally:
            bus.shutdown()

        print(f"\r  Done.                                  \n")

    def print_report(self):
        nodes = sorted(self.nodes.values(), key=lambda r: r.node_id)
        print(f"{'='*58}")
        print(f"  RESULTS — {len(nodes)} node(s)")
        print(f"{'='*58}\n")

        if not nodes:
            print("  No nodes detected.\n")
            print("  Check:")
            print("  • Blue LED slow-blink = running (fast = still booting)")
            print("  • 120Ω terminator at both ends of CAN bus")
            print("  • CAN-H↔CAN-H, CAN-L↔CAN-L wiring")
            print(f"  • Raw sniff:  candump {self.iface}")
            print(f"  • Wrong bitrate?  Try 500k:")
            print(f"      sudo ip link set {self.iface} down")
            print(f"      sudo ip link set {self.iface} up type can bitrate 500000")
            return

        for rec in nodes:
            icon = {"OK": "🟢", "WARNING": "🟡"}.get(rec.health, "🔴")
            print(f"  {icon}  Node {rec.node_id}")
            print(f"       Mode   : {rec.mode}")
            print(f"       Health : {rec.health}")
            print(f"       Uptime : {rec.uptime}s")
            if rec.name: print(f"       Name   : {rec.name}")
            if rec.sw:   print(f"       FW     : {rec.sw}  (0x{rec.vcs:08X})")
            if rec.hw:   print(f"       HW     : {rec.hw}")
            if rec.uid:  print(f"       UID    : {rec.uid}")
            if rec.vendor: print(f"       Vendor : 0x{rec.vendor:04X}")
            print()


def main():
    p = argparse.ArgumentParser(description="DroneCAN scanner — AP_Periph CAN-L4-PWM")
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
