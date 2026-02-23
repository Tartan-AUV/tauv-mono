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
DISCOVERY_TIME = 5.0  # seconds to discover nodes

SWEEP_MAX = 1.0              # max throttle for sweep (0.0 to 1.0)
THRUSTINDEX = 2
THRUSTID = 104
NUM_SWEEPS = 3
SWEEP_STEPS = 200      
SWEEP_STEP_DURATION = 1.3  # seconds to hold each step
SETTLE_TIME = 0.15       # seconds to wait before recording at each step
DATA_FILE = 'thruster_sweep.csv'


LONG_HELP = """
Commands:
    arm                 - Arm the ESCs (enable motor output)    
    disarm              - Disarm the ESCs (disable motor output)
    stop                - Set all throttles to 0
    t <idx> <val>       - Set throttle of ESC at index (0 to
                            ESC_COUNT-1) to value (-1.0 to 1.0) 
    tr <val>           - Set all throttles to value (-1.0 to 1.0)
    ta <v0> <v1> ...    - Set all throttles to specified
                            values (-1.0 to 1.0)        
    telem              - Print latest telemetry from all nodes
    hz <rate>          - Set command rate in Hz (default 50)
    nodes               - List discovered nodes and ESCs
    restart [node_id]  - Restart a node by ID, or all ESCs if
                            no ID given
    sweep <esc_index> <count> - Start a sweep test on the specified ESC index (0 to ESC_COUNT-1)
    quit                - Exit the program
""" 
HELP_TEXT = "Commands: arm, disarm, stop, t <idx> <val>, tr <val>, ta <v0-vN>, telem, hz <rate>, nodes, restart [node_id], sweep <esc_index> <count>, quit"
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

    def set_all_throttles(self, values):
        for i, v in enumerate(values):
            if i < ESC_COUNT:
                self.throttles[i] = max(-1.0, min(1.0, v))

    def stop_all(self):
        self.throttles = [0.0] * ESC_COUNT

    def get_telemetry(self, node_id=None):
        if node_id is not None:
            return self.telemetry.get(node_id)
        return self.telemetry.copy()

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

    def _wait_for_rpm_zero(self):
        print("  Waiting for RPM to reach 0...")
        self.stop_all()
        wait_start = time.time()
        while True:
            if THRUSTID in self.telemetry and self.telemetry[THRUSTID]['rpm'] == 0:
                break
            time.sleep(0.1)
        print(f"  RPM is 0 after {time.time() - wait_start:.1f}s")
        time.sleep(1)

    def _sweep(self, esc_index,SWEEP_MAX=SWEEP_MAX,SWEEP_STEPS=SWEEP_STEPS):
        # 0 -> max, then 0 -> -max
        halves = [
            np.linspace(0, SWEEP_MAX, SWEEP_STEPS // 2),
            np.linspace(0, -SWEEP_MAX, SWEEP_STEPS // 2),
        ]
        buffer = []

        print(f"\nStarting sweep on ESC index {esc_index}")
        print(f"{SWEEP_STEPS} steps x {SWEEP_STEP_DURATION}s = {SWEEP_STEPS * SWEEP_STEP_DURATION:.0f}s total")

        if not self.armed:
            print("Arming for sweep...")
            self.arm()

        for half_idx, gains in enumerate(halves):
            if half_idx == 1:
                print("  Waiting for RPM to reach 0 before negative sweep...")
                self.set_throttle(esc_index, 0.0)
                time.sleep(SETTLE_TIME)
                while True:
                    if THRUSTID in self.telemetry and self.telemetry[THRUSTID]['rpm'] == 0:
                        break
                    time.sleep(0.1)
                print("  RPM is 0, starting negative sweep...")
                time.sleep(1)

            for gain in gains:
                self.set_throttle(esc_index, gain)
                step_start = time.time()
                step_samples = []

                while time.time() - step_start < SWEEP_STEP_DURATION:
                    if THRUSTID in self.telemetry:
                        t = self.telemetry[THRUSTID]
                        step_samples.append({
                            'esc_index': esc_index,
                            'gain':      int(gain * 8191),
                            'timestamp': t['timestamp'],
                            'rpm':       t['rpm'],
                            'voltage':   t['voltage'],
                            'current':   t['current'],
                            'power_pct': t['power_rating_pct'],
                            'is_last':   0,
                        })
                    time.sleep(0.01)

                if step_samples:
                    step_samples[-1]['is_last'] = 1
                    buffer.extend(step_samples)

                if THRUSTID in self.telemetry:
                    t = self.telemetry[THRUSTID]
                    print(f"  gain={gain:+.2f}  rpm={t['rpm']:6d}  V={t['voltage']:.2f}  T={t['temperature_c']:.2f}  A={t['current']:.2f} samples={len(step_samples)}")
                    if t['voltage'] <= 14.0:
                        print("  WARNING: voltage low, stopping sweep")
                        break
                else:
                    print(f"  gain={gain:+.2f}  no telemetry")

        self.set_throttle(esc_index, 0.0)

        import pandas as pd
        df = pd.DataFrame(buffer)
        df = df.drop_duplicates(subset=['timestamp'])

        try:
            existing = pd.read_csv(DATA_FILE)
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_csv(DATA_FILE, index=False)
        print(f"\nSweep done. Saved {len(df)} points to {DATA_FILE}\n> ", end='')
        sys.stdout.flush()

    def _sweep_full(self, esc_index, count):
        steps_phase1 = max(4, int(SWEEP_STEPS * (SWEEP_MAX * 0.6) / SWEEP_MAX))
        steps_phase2 = SWEEP_STEPS

        step_time = SWEEP_STEP_DURATION + SETTLE_TIME
        phase1_time = steps_phase1 * step_time
        phase2_time = steps_phase2 * step_time
        total_time = (phase1_time + phase2_time) * count
        print(f"\nFull sweep: {count} run(s), each ~{phase1_time + phase2_time:.0f}s = ~{total_time:.0f}s total (excludes RPM-zero waits)")
        print(f"Phase 1: {steps_phase1} steps, Phase 2: {steps_phase2} steps")

        for i in range(count):
            self._wait_for_rpm_zero()
            print(f"\n--- Run {i+1}/{count} ---")

            print(f"Phase 1: sweep -{SWEEP_MAX*0.6:.2f} to {SWEEP_MAX*0.6:.2f}  (~{phase1_time:.0f}s)")
            self._sweep(esc_index, SWEEP_MAX=SWEEP_MAX*0.6, SWEEP_STEPS=steps_phase1)

            self._wait_for_rpm_zero()

            print(f"Phase 2: sweep -{SWEEP_MAX:.2f} to {SWEEP_MAX:.2f}  (~{phase2_time:.0f}s)")
            self._sweep(esc_index, SWEEP_MAX=SWEEP_MAX, SWEEP_STEPS=steps_phase2)

            

        print(f"\nAll {count} runs complete.") 
    def handle_command(self, cmd):
        """Process a CLI command. Returns True to continue, False to quit."""
        cmd = cmd.strip()
        if not cmd:
            return True
        
        parts = cmd.lower().split()
        
        if parts[0] in ('quit', 'q'):
            return False
        
        elif parts[0] == 'arm':
            self.arm()
            print("Armed")
        
        elif parts[0] == 'disarm':
            self.disarm()
            print("Disarmed")
        
        elif parts[0] == 'stop':
            self.stop_all()
            print("Stopped")
        
        elif parts[0] == 't' and len(parts) == 3:
            try:
                idx = int(parts[1])
                val = float(parts[2])
                self.set_throttle(idx, val)
                print(f"Throttle[{idx}] = {val}")
            except ValueError:
                print("Usage: t <index> <value>")
        elif parts[0] == 'sweep' and len(parts) == 3:
            try:
                idx = int(parts[1])
                count = int(parts[2])
                import threading
                t = threading.Thread(target=self._sweep_full, args=(idx,count,), daemon=True)
                t.start()
            except ValueError:
                print("Usage: sweep <esc_index> <count>")
        elif parts[0] == 'tr':
            try:
                val = float(parts[1])
                
                self.set_all_throttles([val]*ESC_COUNT)
                
            except ValueError:
                print("Usage: tr <value>")


        elif parts[0] == 'ta':
            try:
                vals = [float(x) for x in parts[1:]]
                if len(vals) != ESC_COUNT:
                    print(f"Need {ESC_COUNT} values")
                else:
                    self.set_all_throttles(vals)
                    print(f"Throttles = {vals}")
            except ValueError:
                print(f"Usage: ta <v0> <v1> ... <v{ESC_COUNT-1}>")
        
        elif parts[0] == 'telem':
            telem = self.get_telemetry()
            if not telem:
                print("No telemetry yet")
            else:
                for nid, t in sorted(telem.items()):
                    print(f"  Node {nid}: {t['rpm']}rpm, {t['voltage']:.1f}V, {t['current']:.1f}A, {t['temperature_c']:.1f}C power_rating_pct={t['power_rating_pct']:.1f}%")
        
        elif parts[0] == 'hz' and len(parts) == 2:
            try:
                self.command_rate_hz = float(parts[1])
                print(f"Command rate = {self.command_rate_hz}Hz")
            except ValueError:
                print("Usage: hz <rate>")
        
        elif parts[0] == 'nodes':
            self.list_nodes()
        
        elif parts[0] == 'restart':
            if len(parts) == 1:
                # Restart all ESCs
                self.restart_all_escs()
            elif len(parts) == 2:
                # Restart specific node
                try:
                    node_id = int(parts[1])
                    print(f"Restarting node {node_id}...")
                    self.restart_node(node_id)
                except ValueError:
                    print("Usage: restart [node_id]")
            else:
                print("Usage: restart [node_id]")
        
        else:
            print(HELP_TEXT)
        
        return True

    def run(self):
        """Main loop with CLI."""
        self.running = True
        period = 1.0 / self.command_rate_hz
        last_cmd_time = 0
        start_time = time.time()
        discovery_done = False
        
        print(f"ESC Controller starting...")
        print(f"Discovering nodes for {DISCOVERY_TIME} seconds...")
        
        try:
            while self.running:
                now = time.time()
                elapsed = now - start_time
                
                # Discovery phase
                if not discovery_done and elapsed >= DISCOVERY_TIME:
                    discovery_done = True
                    print(f"\nDiscovery complete. Found {len(self.nanodrive_escs)} ESCs: {self.nanodrive_escs}")
                    print(f"Running at {self.command_rate_hz}Hz")
                    print(HELP_TEXT)
                    print()
                    sys.stdout.write("> ")
                    sys.stdout.flush()
                
                #Send throttle and arming
                if discovery_done and (now - last_cmd_time >= period):
                    self._send_arming_status()
                    self._send_throttle_command()
                    last_cmd_time = now
                
                # CAN
                try:
                    self.node.spin(timeout=0.001)
                except dronecan.transport.TransferError:
                    pass
                
                #Cool thing chat came up with to handle non-blocking stdin
                if discovery_done and select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        break
                    if not self.handle_command(line):
                        break
                    sys.stdout.write("> ")
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.armed = False
            self.stop_all()
            self._send_arming_status()
            self._send_throttle_command()
            if self.allocator:
                self.allocator.close()
            if self.node_monitor:
                self.node_monitor.close()
            self.node.close()
            print("Shutdown")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    controller = ESCController(command_rate_hz=COMMAND_RATE_HZ)
    controller.run()