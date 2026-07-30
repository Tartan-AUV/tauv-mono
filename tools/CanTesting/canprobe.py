#!/usr/bin/env python3
"""
CAN 2.0 Bus Node Scanner for Jetson Orin
Scans all 127 CANopen node IDs and reports which devices are online.

Usage:
    python3 can_scan.py [--interface can0] [--timeout 0.05] [--retries 2]

Requires:
    pip install python-can
    sudo modprobe can
    sudo modprobe can_raw
    sudo ip link set can0 up type can bitrate 1000000
"""

import can
import time
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional


# CANopen constants
NMT_STATE_REQUEST_BASE = 0x700   # Heartbeat / Node Guarding base COB-ID
SDO_REQUEST_BASE       = 0x600   # SDO request base (0x600 + node_id)
SDO_RESPONSE_BASE      = 0x580   # SDO response base (0x580 + node_id)

# CANopen NMT states
NMT_STATE = {
    0x00: "Boot-up",
    0x04: "Stopped",
    0x05: "Operational",
    0x7F: "Pre-operational",
}

# SDO read request for identity object 0x1000 (device type)
SDO_READ_DEVICE_TYPE = bytes([
    0x40,        # Command: initiate upload request
    0x00, 0x10,  # Index 0x1000 (little-endian)
    0x00,        # Sub-index 0
    0x00, 0x00, 0x00, 0x00  # Padding
])


@dataclass
class NodeInfo:
    node_id: int
    online: bool = False
    method: str = ""
    nmt_state: Optional[str] = None
    device_type: Optional[int] = None
    raw_response: Optional[bytes] = None
    response_time_ms: float = 0.0


def build_argparser():
    p = argparse.ArgumentParser(description="CAN bus node scanner")
    p.add_argument("--interface", default="can0",
                   help="CAN interface name (default: can0)")
    p.add_argument("--bitrate", type=int, default=None,
                   help="Set bitrate if bringing up the interface (optional)")
    p.add_argument("--timeout", type=float, default=0.05,
                   help="Response timeout per node in seconds (default: 0.05)")
    p.add_argument("--retries", type=int, default=2,
                   help="Retries per node on no response (default: 2)")
    p.add_argument("--start", type=int, default=1,
                   help="Start node ID (default: 1)")
    p.add_argument("--end", type=int, default=127,
                   help="End node ID (default: 127)")
    p.add_argument("--method", choices=["nmt", "sdo", "both"], default="both",
                   help="Detection method: nmt heartbeat, sdo ping, or both (default: both)")
    p.add_argument("--verbose", action="store_true",
                   help="Show raw frame data for each response")
    return p


def probe_node_nmt(bus: can.Bus, node_id: int, timeout: float, retries: int) -> NodeInfo:
    """
    Send an NMT state request (Remote Frame on COB-ID 0x700 + node_id).
    A live CANopen node will respond with its heartbeat/state byte.
    """
    info = NodeInfo(node_id=node_id)
    cob_id = NMT_STATE_REQUEST_BASE + node_id

    for attempt in range(retries + 1):
        try:
            req = can.Message(
                arbitration_id=cob_id,
                is_remote_frame=True,
                dlc=1,
                is_extended_id=False,
            )
            t_start = time.monotonic()
            bus.send(req)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                msg = bus.recv(timeout=max(0, deadline - time.monotonic()))
                if msg is None:
                    break
                if msg.arbitration_id == cob_id and not msg.is_remote_frame:
                    elapsed = (time.monotonic() - t_start) * 1000
                    info.online = True
                    info.method = "NMT heartbeat"
                    info.response_time_ms = round(elapsed, 2)
                    info.raw_response = bytes(msg.data)
                    if msg.data:
                        state_byte = msg.data[0] & 0x7F
                        info.nmt_state = NMT_STATE.get(state_byte, f"0x{state_byte:02X}")
                    return info
        except can.CanError:
            pass

    return info


def probe_node_sdo(bus: can.Bus, node_id: int, timeout: float, retries: int) -> NodeInfo:
    """
    Send an SDO upload request for object 0x1000 (device type).
    A live CANopen node will respond on COB-ID 0x580 + node_id.
    """
    info = NodeInfo(node_id=node_id)
    req_cob_id  = SDO_REQUEST_BASE  + node_id
    resp_cob_id = SDO_RESPONSE_BASE + node_id

    for attempt in range(retries + 1):
        try:
            req = can.Message(
                arbitration_id=req_cob_id,
                data=SDO_READ_DEVICE_TYPE,
                is_extended_id=False,
            )
            t_start = time.monotonic()
            bus.send(req)

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                msg = bus.recv(timeout=max(0, deadline - time.monotonic()))
                if msg is None:
                    break
                if msg.arbitration_id == resp_cob_id:
                    elapsed = (time.monotonic() - t_start) * 1000
                    info.online = True
                    info.method = "SDO response"
                    info.response_time_ms = round(elapsed, 2)
                    info.raw_response = bytes(msg.data)
                    # Parse device type if it's a valid SDO upload response (0x43)
                    if len(msg.data) >= 8 and msg.data[0] == 0x43:
                        info.device_type = int.from_bytes(msg.data[4:8], "little")
                    return info
        except can.CanError:
            pass

    return info


def scan_bus(args) -> list[NodeInfo]:
    print(f"\n{'='*60}")
    print(f"  CAN Node Scanner — {args.interface}")
    print(f"  Range  : Node {args.start} – {args.end}")
    print(f"  Method : {args.method.upper()}")
    print(f"  Timeout: {args.timeout*1000:.0f} ms/node  |  Retries: {args.retries}")
    print(f"{'='*60}\n")

    try:
        bus = can.interface.Bus(channel=args.interface, interface="socketcan")
    except Exception as e:
        print(f"[ERROR] Cannot open {args.interface}: {e}")
        print("  → Make sure the interface is up:")
        print(f"      sudo ip link set {args.interface} up type can bitrate 1000000")
        sys.exit(1)

    results: list[NodeInfo] = []
    node_range = range(args.start, args.end + 1)
    total = len(node_range)

    try:
        for i, node_id in enumerate(node_range):
            pct = int((i / total) * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] Node {node_id:3d}/{args.end}", end="", flush=True)

            info = NodeInfo(node_id=node_id)

            if args.method in ("nmt", "both"):
                info = probe_node_nmt(bus, node_id, args.timeout, args.retries)

            if not info.online and args.method in ("sdo", "both"):
                info = probe_node_sdo(bus, node_id, args.timeout, args.retries)

            if info.online:
                results.append(info)

    finally:
        bus.shutdown()

    print(f"\r  [{'█'*40}] Done!                          \n")
    return results


def print_report(results: list[NodeInfo], args):
    online = [r for r in results if r.online]

    print(f"{'='*60}")
    print(f"  SCAN RESULTS — {len(online)} node(s) found online")
    print(f"{'='*60}")

    if not online:
        print("\n  No CANopen nodes responded.\n")
        print("  Troubleshooting tips:")
        print("  • Check CAN bus termination (120Ω at each end)")
        print("  • Verify bitrate matches the devices on the bus")
        print("  • Confirm bus is up: ip link show can0")
        print("  • Try: candump can0   (to see raw traffic)")
        return

    print(f"\n  {'Node':>6}  {'Method':<18}  {'State':<18}  {'RTT':>8}  {'Device Type'}")
    print(f"  {'------':>6}  {'------------------':<18}  {'------------------':<18}  {'--------':>8}  {'------------'}")

    for r in sorted(online, key=lambda x: x.node_id):
        state_str = r.nmt_state or "—"
        dtype_str = f"0x{r.device_type:08X}" if r.device_type is not None else "—"
        rtt_str   = f"{r.response_time_ms:.1f} ms"
        print(f"  {r.node_id:>6}  {r.method:<18}  {state_str:<18}  {rtt_str:>8}  {dtype_str}")

        if args.verbose and r.raw_response:
            hex_bytes = " ".join(f"{b:02X}" for b in r.raw_response)
            print(f"  {'':>6}  Raw: [{hex_bytes}]")

    print(f"\n  Summary: {len(online)}/{args.end - args.start + 1} nodes online\n")


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.start < 1 or args.end > 127 or args.start > args.end:
        print("[ERROR] Node IDs must be between 1 and 127, start ≤ end.")
        sys.exit(1)

    results = scan_bus(args)
    print_report(results, args)


if __name__ == "__main__":
    main()
