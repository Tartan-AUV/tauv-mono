#!/usr/bin/env python3
"""
DroneCAN / UAVCAN v0 Node Scanner
Targets: AP_Periph CAN-L4-PWM (STM32L431, DroneCAN Protocol)

Listens for NodeStatus broadcasts (which AP_Periph sends continuously)
and optionally sends GetNodeInfo requests to enumerate all live nodes.

Usage:
    python3 dronecan_scan.py [--interface can0] [--bitrate 1000000] [--duration 5]

Requires:
    pip install dronecan
    sudo ip link set can0 up type can bitrate 1000000
"""

import time
import argparse
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    import dronecan
except ImportError:
    print("[ERROR] dronecan package not found.")
    print("  Install it:  pip install dronecan")
    sys.exit(1)


# DroneCAN NodeStatus health values
HEALTH = {0: "OK", 1: "WARNING", 2: "ERROR", 3: "CRITICAL"}

# DroneCAN NodeStatus mode values
MODE = {
    0: "OPERATIONAL",
    1: "INITIALIZATION",
    2: "MAINTENANCE",
    3: "SOFTWARE_UPDATE",
    7: "OFFLINE",
}


@dataclass
class NodeRecord:
    node_id: int
    health: str = "?"
    mode: str = "?"
    uptime_sec: float = 0.0
    vendor_specific_status: int = 0
    # From GetNodeInfo response
    name: str = ""
    hw_version: str = ""
    sw_version: str = ""
    sw_vcs_commit: int = 0
    uid: str = ""
    # Tracking
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    info_requested: bool = False
    info_received: bool = False


class DroneCAN_Scanner:
    def __init__(self, interface: str, bitrate: int, duration: float, verbose: bool):
        self.interface = interface
        self.bitrate   = bitrate
        self.duration  = duration
        self.verbose   = verbose
        self.nodes: dict[int, NodeRecord] = {}
        self._lock = threading.Lock()
        self._node = None

    # ------------------------------------------------------------------ #
    #  Handlers                                                            #
    # ------------------------------------------------------------------ #

    def _on_node_status(self, event):
        msg = event.message
        nid = event.transfer.source_node_id
        now = time.monotonic()

        health = HEALTH.get(msg.health, f"0x{msg.health:X}")
        mode   = MODE.get(msg.mode, f"0x{msg.mode:X}")

        with self._lock:
            if nid not in self.nodes:
                self.nodes[nid] = NodeRecord(node_id=nid, first_seen=now)
                print(f"\r  [+] Node {nid:3d} appeared  |  {mode:<16}  health: {health}")

            rec = self.nodes[nid]
            rec.health   = health
            rec.mode     = mode
            rec.uptime_sec = msg.uptime_sec
            rec.vendor_specific_status = msg.vendor_specific_status_code
            rec.last_seen = now

            # Request GetNodeInfo once per node
            if not rec.info_requested:
                rec.info_requested = True
                self._request_node_info(nid)

    def _on_node_info(self, event):
        if event is None:
            return   # timeout / no response
        msg  = event.message
        nid  = event.transfer.source_node_id

        with self._lock:
            if nid not in self.nodes:
                return
            rec = self.nodes[nid]
            rec.info_received = True

            try:
                rec.name = "".join(chr(c) for c in msg.name.encode())
            except Exception:
                rec.name = str(msg.name)

            try:
                rec.hw_version = f"{msg.hardware_version.major}.{msg.hardware_version.minor}"
                rec.uid = msg.hardware_version.unique_id.tolist()
                rec.uid = " ".join(f"{b:02X}" for b in rec.uid)
            except Exception:
                pass

            try:
                rec.sw_version    = f"{msg.software_version.major}.{msg.software_version.minor}"
                rec.sw_vcs_commit = msg.software_version.vcs_commit
            except Exception:
                pass

            if self.verbose:
                print(f"\r  [i] Node {nid:3d} info: {rec.name}  SW {rec.sw_version}  HW {rec.hw_version}")

    def _request_node_info(self, node_id: int):
        """Fire-and-forget GetNodeInfo request."""
        try:
            self._node.request(
                dronecan.uavcan.protocol.GetNodeInfo.Request(),
                node_id,
                self._on_node_info,
                timeout=2.0,
                priority=dronecan.transport.TRANSFER_PRIORITY_LOW,
            )
        except Exception as e:
            if self.verbose:
                print(f"\r  [!] GetNodeInfo failed for node {node_id}: {e}")

    # ------------------------------------------------------------------ #
    #  Main scan loop                                                      #
    # ------------------------------------------------------------------ #

    def run(self):
        print(f"\n{'='*62}")
        print(f"  DroneCAN Node Scanner")
        print(f"  Interface : {self.interface}  |  Bitrate: {self.bitrate:,} bps")
        print(f"  Listening for {self.duration}s …")
        print(f"{'='*62}\n")

        # Create a local DroneCAN node (node ID 127 = ground station / monitor)
        try:
            self._node = dronecan.make_node(
                self.interface,
                node_id=127,
                bitrate=self.bitrate,
            )
        except Exception as e:
            print(f"[ERROR] Cannot open {self.interface}: {e}")
            print(f"  → Bring up the interface first:")
            print(f"      sudo ip link set {self.interface} up type can bitrate {self.bitrate}")
            sys.exit(1)

        # Subscribe to NodeStatus (broadcast ~1 Hz by every DroneCAN node)
        self._node.add_handler(
            dronecan.uavcan.protocol.NodeStatus,
            self._on_node_status,
        )

        t_start  = time.monotonic()
        t_end    = t_start + self.duration
        spinner  = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        spin_i   = 0

        try:
            while time.monotonic() < t_end:
                remaining = t_end - time.monotonic()
                with self._lock:
                    n_found = len(self.nodes)
                print(
                    f"\r  {spinner[spin_i % len(spinner)]}  "
                    f"{remaining:4.1f}s remaining  |  {n_found} node(s) found",
                    end="", flush=True,
                )
                spin_i += 1
                self._node.spin(timeout=0.1)

        except KeyboardInterrupt:
            print("\n  [!] Interrupted by user.")
        finally:
            self._node.close()

        print(f"\r  ✓  Scan complete.                                      \n")

    # ------------------------------------------------------------------ #
    #  Report                                                              #
    # ------------------------------------------------------------------ #

    def print_report(self):
        with self._lock:
            nodes = list(self.nodes.values())

        print(f"{'='*62}")
        print(f"  RESULTS — {len(nodes)} DroneCAN node(s) found")
        print(f"{'='*62}\n")

        if not nodes:
            print("  No nodes responded.\n")
            print("  Troubleshooting:")
            print("  • AP_Periph blue LED should be slow-blinking (running)")
            print("  • Fast-blinking blue = still booting, wait a moment")
            print("  • Check CAN termination: 120Ω resistors at both ends")
            print(f"  • Verify bitrate: candump {self.interface} and look for traffic")
            print("  • AP_Periph default CAN bitrate [118;1:3uis usually 1 Mbps")
            print("  • Try: python3 dronecan_scan.py --bitrate 500000")
            return

        for rec in sorted(nodes, key=lambda r: r.node_id):
            status_icon = "🟢" if rec.health == "OK" else "🟡" if rec.health == "WARNING" else "🔴"
            print(f"  {status_icon}  Node ID : {rec.node_id}")
            print(f"       Mode    : {rec.mode}")
            print(f"       Health  : {rec.health}")
            print(f"       Uptime  : {rec.uptime_sec}s")

            if rec.info_received:
                print(f"       Name    : {rec.name}")
                print(f"       SW ver  : {rec.sw_version}  (commit: 0x{rec.sw_vcs_commit:08X})")
                print(f"       HW ver  : {rec.hw_version}")
                if rec.uid:
                    print(f"       UID     : {rec.uid}")
            else:
                print(f"       Info    : (GetNodeInfo not yet received)")

            vs = rec.vendor_specific_status
            if vs:
                print(f"       Vendor  : 0x{vs:04X}", end="")
                # AP_Periph encodes fault flags in vendor status
                flags = []
                if vs & 0x01: flags.append("CAN_INIT_FAIL")
                if vs & 0x02: flags.append("SERVO_INIT_FAIL")
                if vs & 0x04: flags.append("BARO_INIT_FAIL")
                if flags:
                    print(f"  [{', '.join(flags)}]", end="")
                print()

            print()

        # Quick summary table
        print(f"  {'ID':>4}  {'Name':<30}  {'Mode':<16}  {'Health':<10}  {'SW'}")
        print(f"  {'----':>4}  {'------------------------------':<30}  {'----------------':<16}  {'----------':<10}  {'------'}")
        for rec in sorted(nodes, key=lambda r: r.node_id):
            name = rec.name if rec.name else "(unknown)"
            sw   = rec.sw_version if rec.sw_version else "?"
            print(f"  {rec.node_id:>4}  {name:<30}  {rec.mode:<16}  {rec.health:<10}  {sw}")
        print()


def main():
    p = argparse.ArgumentParser(
        description="DroneCAN node scanner for AP_Periph CAN-L4-PWM and similar devices"
    )
    p.add_argument("--interface", default="can0",
                   help="SocketCAN interface (default: can0)")
    p.add_argument("--bitrate", type=int, default=1000000,
                   help="CAN bitrate in bps (default: 1000000)")
    p.add_argument("--duration", type=float, default=5.0,
                   help="How long to listen in seconds (default: 5)")
    p.add_argument("--verbose", action="store_true",
                   help="Print each GetNodeInfo response as it arrives")
    args = p.parse_args()

    scanner = DroneCAN_Scanner(
        interface=args.interface,
        bitrate=args.bitrate,
        duration=args.duration,
        verbose=args.verbose,
    )
    scanner.run()
    scanner.print_report()


if __name__ == "__main__":
    main()
