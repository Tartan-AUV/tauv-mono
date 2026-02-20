#!/usr/bin/env python3
"""
DroneCAN ESC Controller
Chat is awesome for cleaning up my code and writing a cli!
"""

import time
import sys
import select
import dronecan
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

INTERFACE = 'can1'
NODE_ID = 12
BITRATE = 1000000
ESC_COUNT = 8

COMMAND_RATE_HZ = 50
DNA_DB_PATH = "./dronecan_dna.db"
DISCOVERY_TIME = 8.0 

THRUSTINDEX = 2
THRUSTID = 104
NUM_SWEEPS = 3
SWEEP_STEPS = 50        # number of gain steps across -1 to 1
SWEEP_STEP_DURATION = 1.0  # seconds to hold each step
SETTLE_TIME = 0.3       # seconds to wait before recording at each step
DATA_FILE = 'thruster_sweep.parquet'


# =============================================================================


class ESCController:
    def __init__(self, command_rate_hz=COMMAND_RATE_HZ):
        self.running = False
        self.command_rate_hz = command_rate_hz

        self.node_monitor = None
        self.allocator = None
        
        self.throttles = [0.0] * ESC_COUNT
        self.telemetry = {}
        self.armed = False
        
        # Discovered nodes
        self.discovered_nodes = {}  # node_id -> node_info
        self.nanodrive_escs = []    # list of nanodrive node IDs
        
        self.on_node_status = None
        
        self._init_node()

    def _init_node(self):
        node_info = dronecan.uavcan.protocol.GetNodeInfo.Response()
        node_info.name = 'esc_controller'
        node_info.software_version.major = 1
        
        self.node = dronecan.make_node(
            INTERFACE,
            node_id=NODE_ID,
            bitrate=BITRATE,
            node_info=node_info
        )
        self.node.mode = dronecan.uavcan.protocol.NodeStatus().MODE_OPERATIONAL

        #DNA server
        self.node_monitor = dronecan.app.node_monitor.NodeMonitor(self.node)
        self.allocator = dronecan.app.dynamic_node_id.CentralizedServer(
            self.node, self.node_monitor, database_storage=DNA_DB_PATH
        )

        self.node.add_handler(dronecan.uavcan.equipment.esc.Status, self._on_esc_status)
        self.node.add_handler(dronecan.uavcan.protocol.NodeStatus, self._on_node_status)

    def _on_esc_status(self, event):
        msg = event.message
        node_id = event.transfer.source_node_id
        
        self.telemetry[node_id] = {
            'timestamp': time.time(),
            'error_count': msg.error_count,
            'voltage': msg.voltage,
            'current': msg.current,
            'temperature': msg.temperature,
            'temperature_c': msg.temperature - 273.15 if msg.temperature > 0 else 0,
            'rpm': msg.rpm,
            'power_rating_pct': msg.power_rating_pct,
            'esc_index': msg.esc_index,
        }
        
        
        self.on_telemetry(node_id, self.telemetry[node_id])

    def _on_node_status(self, event):
        msg = event.message
        node_id = event.transfer.source_node_id
        
        # Discover new nodes
        if node_id not in self.discovered_nodes:
            self.discovered_nodes[node_id] = None
            self._request_node_info(node_id)
    
        #!TODO add a callback for node status if needed

    def _request_node_info(self, node_id):
        """Request GetNodeInfo to identify the node."""
        def callback(event):
            if event and event.response:
                name = bytes(event.response.name).decode().rstrip('\x00')
                self.discovered_nodes[node_id] = {
                    'name': name,
                    'sw_version': f"{event.response.software_version.major}.{event.response.software_version.minor}",
                    'hw_version': f"{event.response.hardware_version.major}.{event.response.hardware_version.minor}",
                }
                print(f"Found node {node_id}: {name}")
                
                # Track nanodrive ESCs
                if 'nanodrive' in name.lower():
                    if node_id not in self.nanodrive_escs:
                        self.nanodrive_escs.append(node_id)
                        self.nanodrive_escs.sort()
        
        self.node.request(dronecan.uavcan.protocol.GetNodeInfo.Request(), node_id, callback)

    def _send_throttle_command(self):
        raw_values = [int(t * 8191) for t in self.throttles]
        msg = dronecan.uavcan.equipment.esc.RawCommand(cmd=raw_values)
        self.node.broadcast(msg)

    def _send_arming_status(self):
        status = 255 if self.armed else 0
        msg = dronecan.uavcan.equipment.safety.ArmingStatus(status=status)
        self.node.broadcast(msg)

    def on_telemetry(self, node_id, data):
        pass#nothing to do with telm now
   
    def set_throttle(self, index, value):
        if 0 <= index < ESC_COUNT:
            self.throttles[index] = max(-1.0, min(1.0, value))

    def stop_all(self):
        self.throttles = [0.0] * ESC_COUNT

    def arm(self):
        self.armed = True

    def disarm(self):
        self.armed = False
        self.stop_all()

    def restart_node(self, node_id):
        """Send restart command to a node."""
        def callback(event):
            if event and event.response and event.response.ok:
                print(f"  Node {node_id}: restart accepted")
            else:
                print(f"  Node {node_id}: restart failed/timeout")
        
        req = dronecan.uavcan.protocol.RestartNode.Request()
        req.magic_number = req.MAGIC_NUMBER  # 0xACCE551B1E
        self.node.request(req, node_id, callback)

    def restart_all_escs(self):
        """Restart all discovered nanodrive ESCs."""
        if not self.nanodrive_escs:
            print("No nanodrive ESCs discovered")
            return
        
        print(f"Restarting {len(self.nanodrive_escs)} ESCs...")
        for node_id in self.nanodrive_escs:
            self.restart_node(node_id)

    def list_nodes(self):
        """Print all discovered nodes."""
        print(f"Discovered {len(self.discovered_nodes)} nodes:")
        for node_id, info in sorted(self.discovered_nodes.items()):
            if info:
                esc_marker = " [ESC]" if node_id in self.nanodrive_escs else ""
                print(f"  Node {node_id}: {info['name']}{esc_marker}")
            else:
                print(f"  Node {node_id}: (info pending)")
        print(f"Nanodrive ESCs: {self.nanodrive_escs}")

    def run(self, num_sweeps=1):
        self.running = True
        start_time = time.time()
        discovery_done = False

        # Sweep state machine
        sweep_index = 0
        gain_index = 0
        step_start = None
        between_sweeps_start = None
        gains = np.linspace(-1.0, 1.0, SWEEP_STEPS)
        buffer = []
        sweep_done = False

        last_cmd_time = 0
        CMD_PERIOD = 1.0 / self.command_rate_hz

        print(f"Discovering nodes for {DISCOVERY_TIME} seconds...")

        try:
            while self.running:
                now = time.time()

                # --- CAN spin (always) ---
                try:
                    self.node.spin(timeout=0.001)
                except dronecan.transport.TransferError:
                    pass

                # --- Send commands on timer (always after armed) ---
                if self.armed and (now - last_cmd_time >= CMD_PERIOD):
                    self._send_arming_status()
                    self._send_throttle_command()
                    last_cmd_time = now

                # --- Discovery ---
                if not discovery_done and (now - start_time >= DISCOVERY_TIME):
                    discovery_done = True
                    print(f"Discovery complete. ESCs: {self.nanodrive_escs}")
                    self.arm()
                
                    print(f"\nSweep 1/{num_sweeps}")
                    self.set_throttle(THRUSTINDEX, gains[0])
                    step_start = now

                # --- Sweep state machine ---
                if not discovery_done or sweep_done:
                    continue

                # Between sweeps cooldown
                if between_sweeps_start is not None:
                    if now - between_sweeps_start < 5.0:
                        continue
                    else:
                        between_sweeps_start = None
                        print(f"\nSweep {sweep_index + 1}/{num_sweeps}")
                        gain_index = 0
                        self.set_throttle(THRUSTINDEX, gains[0])
                        step_start = now

                # Step timer
                if now - step_start < SWEEP_STEP_DURATION:
                    continue

                # Step complete — record
                gain = gains[gain_index]
                if THRUSTID in self.telemetry:
                    t = self.telemetry[THRUSTID]
                    buffer.append({
                        'gain': gain,
                        'power_pct': t['power_rating_pct'],
                        'rpm': t['rpm'],
                        'voltage': t['voltage'],
                        'current': t['current'],
                    })
                    print(f"  gain={gain:+.2f}  rpm={t['rpm']:6d}  V={t['voltage']:.2f}")
                    if t['voltage'] <= 13.5:
                        print("  WARNING: Voltage low, stopping sweep")
                        gain_index = len(gains)  # force end
                else:
                    print(f"  gain={gain:+.2f}  no telemetry yet")

                gain_index += 1

                if gain_index < len(gains):
                    # Next step
                    self.set_throttle(THRUSTINDEX, gains[gain_index])
                    step_start = now
                else:
                    # Sweep complete — save
                    self.set_throttle(THRUSTINDEX, 0.0)
                    self._save_buffer(buffer)
                    buffer = []
                    sweep_index += 1

                    if sweep_index < num_sweeps:
                        print(f"Sweep {sweep_index} done. Cooling down 5s...")
                        between_sweeps_start = now
                    else:
                        print("All sweeps complete.")
                        self.disarm()
                        sweep_done = True
                        self.running = False

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.disarm()
            self._send_arming_status()
            self._send_throttle_command()
            if self.allocator: self.allocator.close()
            if self.node_monitor: self.node_monitor.close()
            self.node.close()
            print("Shutdown")

    def _save_buffer(self, buffer):
        import pandas as pd
        df = pd.DataFrame(buffer)
        try:
            existing = pd.read_parquet(DATA_FILE)
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_parquet(DATA_FILE, index=False)
        print(f"Saved {len(buffer)} points to {DATA_FILE}")
   
  
# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    controller = ESCController(command_rate_hz=COMMAND_RATE_HZ)
    controller.run(NUM_SWEEPS)