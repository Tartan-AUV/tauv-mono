import dronecan, time, math
from dronecan import uavcan

# Initializing a DroneCAN node instance.
node = dronecan.make_node("can1", node_id=127, bitrate=1000000)
# Initializing a node monitor, so we can see what nodes are online
node_monitor = dronecan.app.node_monitor.NodeMonitor(node)
dynamic_node_id_allocator = dronecan.app.dynamic_node_id.CentralizedServer(node, node_monitor)
def detect_esc_nodes():
    esc_nodes = set()
    handle = node.add_handler(uavcan.equipment.esc.Status, lambda event: esc_nodes.add(event.transfer.source_node_id))
    try:
        node.spin(timeout=3)            # Collecting ESC status messages, thus determining which nodes are ESC
    finally:
        handle.remove()

    return esc_nodes
print(detect_esc_nodes())
# callback for printing all messages in human-readable YAML format.
# node.add_handler(None, lambda msg: print(dronecan.to_yaml(msg)))

# Running the node until the application is terminated or until first error.
try:
    node.spin()
except KeyboardInterrupt:
    pass
