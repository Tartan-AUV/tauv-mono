#!/usr/bin/env python3
"""
DroneCAN ESC Parameter Fetcher/Setter with DNA Server
"""

import time
import os
import dronecan

# =============================================================================
# CONFIGURATION - Edit these variables
# =============================================================================

INTERFACE = 'can1'
NODE_ID = 10     
BITRATE = 1000000 
DISCOVERY_TIME = 5.0 
DNA_DB_PATH = "./dronecan_dna.db"

# Parameter file to apply (set to None to skip, or path to file)
# File format: one "param_name=value" per line, lines starting with # are comments
PARAM_FILE = "./cleanerparams.txt"  # Example: "./esc_params.txt"
# =============================================================================


# Save params to flash after setting
SAVE_AFTER_SET = True
# Request timeout (seconds) and retries
REQUEST_TIMEOUT = 1.0
MAX_RETRIES = 3

# =============================================================================


class ESCParamManager:
    def __init__(self):
        self.discovered_nodes = {}
        self.node_parameters = {}
        self.params_to_set = {}
        
        # Queue for sequential operations
        self.pending_fetch = []
        self.pending_set = []  # (node_id, name, value)
        self.pending_save = []
        self.pending_verify = []
        self.current_fetch_node = None
        self.current_fetch_idx = 0
        self.busy = False
        self.verify_mode = False
        self.verify_results = {}  # node_id -> {param: (expected, actual, match)}
        self.retry_count = 0
        self.max_retries = 3
        
        self._init_node()
        self._load_param_file()

    def _init_node(self):
        node_info = dronecan.uavcan.protocol.GetNodeInfo.Response()
        node_info.name = 'Jetson'
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
        # Handler for node status messages to discover nodes
        self.node.add_handler(dronecan.uavcan.protocol.NodeStatus, self._on_node_status)

    def _load_param_file(self):
        if not PARAM_FILE or not os.path.exists(PARAM_FILE):
            return
        
        with open(PARAM_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                    
                parts = line.split('=', 1)
                if len(parts) != 2:
                    continue
                    
                name = parts[0].strip()
                value_str = parts[1].strip()
                
                # Parse value
                try:
                    if '.' in value_str:
                        value = float(value_str)
                    elif value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str  # Keep as string
                
                self.params_to_set[name] = value
        
        if self.params_to_set:
            print(f"Loaded {len(self.params_to_set)} params from {PARAM_FILE}")
            # for name, value in self.params_to_set.items():
            #     print(f"  {name} = {value} ({type(value).__name__})")

    def _on_node_status(self, event):
        node_id = event.transfer.source_node_id
        if node_id not in self.discovered_nodes and 99 <= node_id <= 100:
            self.discovered_nodes[node_id] = None
            print(f"Found node {node_id}")
            self._request_node_info(node_id)

        

    def _request_node_info(self, node_id):
        def callback(event):
            if event:
                self.discovered_nodes[node_id] = event.response
                name = bytes(event.response.name).decode().rstrip('\x00')
                print(f"Node {node_id}: {name}")
        
        self.node.request(dronecan.uavcan.protocol.GetNodeInfo.Request(), node_id, callback)

    def _get_param_value(self, value_union):
        if value_union is None:
            return None
        
        for attr in ['integer_value', 'real_value', 'boolean_value', 'string_value']:
            val = getattr(value_union, attr, None)
            if val is not None:
                if isinstance(val, (bytes, bytearray)):
                    return bytes(val).decode().rstrip('\x00')
                return val
        return None

    def _process_next(self):
        """Process next item in queue."""
        if self.busy:
            return
        
        # Priority: fetch -> set -> save -> verify
        if self.current_fetch_node is not None:
            self._do_fetch_next()
        elif self.pending_fetch:
            node_id = self.pending_fetch.pop(0)
            self.node_parameters[node_id] = {}
            self.current_fetch_node = node_id
            self.current_fetch_idx = 0
            self._do_fetch_next()
        elif self.pending_set:
            node_id, name, value = self.pending_set.pop(0)
            self._do_set_param(node_id, name, value)
        elif self.pending_save:
            node_id = self.pending_save.pop(0)
            self._do_save_params(node_id)
        elif self.pending_verify:
            node_id = self.pending_verify.pop(0)
            self._start_verify(node_id)
        elif self.verify_mode and not self.pending_fetch:
            # All verification done
            self._print_verify_results()

    def _do_fetch_next(self):
        """Fetch next parameter from current node."""
        self.busy = True
        node_id = self.current_fetch_node
        idx = self.current_fetch_idx

        def callback(event):
            self.busy = False
            if event and event.response:
                self.retry_count = 0  # Reset on success
                name = bytes(event.response.name).decode().rstrip('\x00')
                if name:
                    value = self._get_param_value(event.response.value)
                    self.node_parameters[node_id][name] = value
                    
                    if self.verify_mode:
                        # Check against expected
                        if name in self.params_to_set:
                            expected = self.params_to_set[name]
                            match = self._values_match(expected, value)
                            self.verify_results[node_id][name] = (expected, value, match)
                            status = "OK" if match else "MISMATCH"
                            print(f"  [{node_id}] VERIFY {name}: expected={expected}, got={value} [{status}]")
                    else:
                        print(f"  [{node_id}] {name} = {value}")
                    
                    self.current_fetch_idx += 1
                else:
                    # Done with this node
                    count = len(self.node_parameters[node_id])
                    if self.verify_mode:
                        print(f"Node {node_id}: verification complete")
                    else:
                        print(f"Node {node_id}: {count} params")
                        
                        # Queue params to set for this node
                        if self.params_to_set:
                            for pname, pvalue in self.params_to_set.items():
                                if pname in self.node_parameters[node_id]:
                                    self.pending_set.append((node_id, pname, pvalue))
                            if SAVE_AFTER_SET:
                                self.pending_save.append(node_id)
                                self.pending_verify.append(node_id)
                    
                    self.current_fetch_node = None
                
                # Small delay between fetches
                self.node.defer(0.02, self._process_next)
            else:
                # Timeout - retry or move on
                self.retry_count += 1
                if self.retry_count < self.max_retries:
                    print(f"  [{node_id}] idx {idx} timeout, retry {self.retry_count}/{self.max_retries}")
                    self.node.defer(0.1, self._process_next)  # Wait before retry
                else:
                    print(f"  [{node_id}] idx {idx} failed after {self.max_retries} retries, skipping")
                    self.retry_count = 0
                    self.current_fetch_idx += 1  # Skip this param and continue
                    self.node.defer(0.1, self._process_next)

        req = dronecan.uavcan.protocol.param.GetSet.Request()
        req.index = idx
        self.node.request(req, node_id, callback, timeout=1.0)  # Longer timeout

    def _do_set_param(self, node_id, name, value):
        """Set a single parameter."""
        self.busy = True

        def callback(event):
            self.busy = False
            if event and event.response:
        
                new_val = self._get_param_value(event.response.value)
                print(f"  [{node_id}] SET {name} = {new_val}")
            else:
                print(f"  [{node_id}] SET {name} TIMEOUT")
            
            # Small delay between sets
            self.node.defer(0.05, self._process_next)

        req = dronecan.uavcan.protocol.param.GetSet.Request()
        req.name = name
        if req.name == "DRONECAN_INDEX":
            req.value.integer_value = node_id-100
        elif isinstance(value, bool):
            req.value.boolean_value = value
        elif isinstance(value, int):
            req.value.integer_value = value
        elif isinstance(value, float):
            req.value.real_value = value
        else:
            req.value.string_value = str(value).encode()

        self.node.request(req, node_id, callback, timeout=2.0)

    def _do_save_params(self, node_id):
        """Save params to flash."""
        self.busy = True

        def callback(event):
            self.busy = False
            if event and event.response and event.response.ok:
                print(f"  [{node_id}] SAVED to flash")
            else:
                print(f"  [{node_id}] SAVE FAILED")
            self.node.defer(0.5, self._process_next)  # Wait after save

        req = dronecan.uavcan.protocol.param.ExecuteOpcode.Request()
        req.opcode = req.OPCODE_SAVE
        self.node.request(req, node_id, callback, timeout=5.0)  # Longer timeout for save

    def _start_verify(self, node_id):
        """Start verification for a node by re-reading all params."""
        print(f"\nVerifying node {node_id}...")
        self.verify_mode = True
        self.verify_results[node_id] = {}
        self.node_parameters[node_id] = {}
        self.current_fetch_node = node_id
        self.current_fetch_idx = 0
        self._process_next()

    def _values_match(self, expected, actual):
        """Check if values match (with tolerance for floats)."""
        if isinstance(expected, float) or isinstance(actual, float):
            try:
                return abs(float(expected) - float(actual)) < 0.001
            except (ValueError, TypeError):
                return False
        return expected == actual

    def _print_verify_results(self):
        """Print verification summary."""
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        
        all_ok = True
        for node_id in sorted(self.verify_results.keys()):
            results = self.verify_results[node_id]
            passed = sum(1 for _, _, match in results.values() if match)
            failed = len(results) - passed
            
            print(f"\nNode {node_id}: {passed}/{len(results)} params OK")
            
            if failed > 0:
                all_ok = False
                print("  MISMATCHES:")
                for name, (expected, actual, match) in results.items():
                    if not match:
                        print(f"    {name}: expected={expected}, got={actual}")
        
        print("\n" + "=" * 60)
        if all_ok:
            print("ALL PARAMETERS VERIFIED SUCCESSFULLY")
        else:
            print("SOME PARAMETERS FAILED VERIFICATION")
        print("=" * 60)

    def start_fetch_all(self):
       #create fetch queue
        for node_id in sorted(self.discovered_nodes.keys()):
            if node_id not in self.node_parameters: # if we haven't fetched yet add to queue
                self.pending_fetch.append(node_id)
        self._process_next()

    def run(self, duration=None):
        start = time.time()
        discovery_done = False

        try:
            while True:
                try:
                    self.node.spin(timeout=0.1)
                except dronecan.transport.TransferError as e:
                    print(f"TransferError: {e}")
                    continue
                except Exception as e:
                    if 'CAN' in str(type(e).__name__):
                        print(f"CAN Error: {e}")
                        continue
                    raise

                elapsed = time.time() - start
                #wait until we find all the nodes before starting fetch
                if not discovery_done and elapsed >= DISCOVERY_TIME:
                    discovery_done = True
                    print(f"\nDiscovered {len(self.discovered_nodes)} nodes")
                    print("Fetching params (sequential)...\n")
                    self.start_fetch_all()

                if duration and elapsed >= duration:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        print(f"\nDone.")
        self.allocator.close()
        self.node_monitor.close()
        self.node.close()


if __name__ == '__main__':
    mgr = ESCParamManager()
    mgr.run()