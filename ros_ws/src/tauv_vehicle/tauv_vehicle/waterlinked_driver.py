import rclpy
from rclpy.node import Node
import asyncio
import json
import time
import numpy as np
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tauv_msgs.msg import WaterlinkedDvlFrame

class WaterlinkedDriver(Node):

    JSON_PROTOCOL_VERSION = "json_v3.1"
    
    # Float64 limits
    FLOAT64_MAX = np.finfo(np.float64).max
    FLOAT64_MIN = np.finfo(np.float64).min
    
    # Reasonable threshold for covariance values (adjust as needed)
    COVARIANCE_WARNING_THRESHOLD = 1e6  # Warn when covariance exceeds this
    
    def __init__(self):
        super().__init__("waterlinked_dvl")
        self.declare_parameter("speed_of_sound", 1481.0)
        self.declare_parameter("mounting_rotation_offset_deg", 0.0)
        self.declare_parameter("range_mode", "auto")
        self.declare_parameter("periodic_cycling_enable", True)
        self.declare_parameter("address", "10.0.0.22")
        self.declare_parameter("port", 16171)
        self.declare_parameter("response_timeout", 2.0)
        self.declare_parameter("connection_timeout", 10.0)

        self.reader = None
        self.writer = None
        self.awaiting_ack = False
        
        # Track excessive covariance warnings
        self.excessive_covariance_count = 0
        self.last_covariance_warning_time = 0

        self.packet_publisher = self.create_publisher(WaterlinkedDvlFrame, "dvl_frame", 10)

    async def connect(self):
        address = self.get_parameter("address").get_parameter_value().string_value
        port = self.get_parameter("port").get_parameter_value().integer_value

        try:
            self.reader, self.writer = await asyncio.open_connection(address, port)
            self.get_logger().info(f"Connected to {address}:{port}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to connect to {address}:{port}: {e}")
            return False

    async def disconnect(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            self.get_logger().info("Disconnected from server")

    async def upload_config(self):
        if not self.writer:
            self.get_logger().error("Not connected. Call connect() first.")
            return

        config = {
            "command":"set_config",
            "parameters": {
                "speed_of_sound": self.get_parameter("speed_of_sound").get_parameter_value().double_value,
                "acoustic_enabled": True,
                "dark_mode_enabled": False,
                "mounting_rotation_offset": self.get_parameter("mounting_rotation_offset_deg").get_parameter_value().double_value,
                "range_mode": self.get_parameter("range_mode").get_parameter_value().string_value,
                "periodic_cycling_enabled": self.get_parameter("periodic_cycling_enable").get_parameter_value().bool_value
            }
        }

        config_json = json.dumps(config)
        response_timeout = self.get_parameter("response_timeout").get_parameter_value().double_value
        connection_timeout = self.get_parameter("connection_timeout").get_parameter_value().double_value

        start_time = time.monotonic()

        while time.monotonic() - start_time < connection_timeout:
            try:
                self.awaiting_ack = True
                self.writer.write(config_json.encode('utf-8') + b'\n')
                await self.writer.drain()
                self.get_logger().info("Config sent, awaiting response...")

                response_line = await asyncio.wait_for(self.reader.readline(), timeout=response_timeout)
                response = response_line.decode('utf-8').strip()
                self.awaiting_ack = False

                try:
                    response_data = json.loads(response)
                    if response_data.get("response_to") == "set_config" :
                        if response_data.get("success") is True:
                            self.get_logger().info("Configuration upload successful.")
                            return
                        else:
                            self.get_logger().warn("Unexpected or failed config response.")
                except json.JSONDecodeError:
                    self.get_logger().error("Failed to parse JSON response.")

            except asyncio.TimeoutError:
                self.get_logger().warn("Timeout waiting for config response, retrying...")
            except Exception as e:
                self.get_logger().error(f"Error during config upload: {e}")
                self.awaiting_ack = False
                return

            

        self.awaiting_ack = False
        self.get_logger().error("Failed to receive valid response within connection timeout.")

    async def listen_for_packets(self):
        if not self.reader:
            self.get_logger().error("Cannot listen for packets: not connected")
            return
        
        while rclpy.ok():
            try:
                line = await self.reader.readline()
                if not self.awaiting_ack:
                    self.handle_packet(line)
            except Exception as e:
                self.get_logger().error(f"Error reading packet: {e}")
                break

    def handle_packet(self, data):
        try:
            message = data.decode('utf-8').strip()
            packet = json.loads(message)

            if packet.get("format") != self.JSON_PROTOCOL_VERSION:
                self.get_logger().warn(f"Unexpected JSON format version: {packet.get('format')}")

            if packet.get("type") == "velocity":
                msg = WaterlinkedDvlFrame()
                msg.time = float(packet["time"])
                msg.vx = float(packet["vx"])
                msg.vy = float(packet["vy"])
                msg.vz = float(packet["vz"])
                msg.fom = float(packet["fom"])
                msg.altitude = float(packet["altitude"])
                msg.velocity_valid = bool(packet["velocity_valid"])
                msg.status = int(packet["status"])
                msg.time_of_validity = int(packet["time_of_validity"])
                msg.time_of_transmission = int(packet["time_of_transmission"])
                
                # Process covariance matrix with overflow handling
                covariance_matrix = packet["covariance"]
                flat_cov = []
                has_excessive_values = False
                max_cov_value = 0
                
                for row in covariance_matrix:
                    for val in row:
                        # Track maximum absolute value
                        abs_val = abs(val) if not np.isinf(val) else float('inf')
                        max_cov_value = max(max_cov_value, abs_val)
                        
                        # Check if value exceeds warning threshold
                        if abs_val > self.COVARIANCE_WARNING_THRESHOLD:
                            has_excessive_values = True
                        
                        # Clamp to float64 range
                        if val > self.FLOAT64_MAX or np.isinf(val):
                            clamped_val = self.FLOAT64_MAX
                        elif val < self.FLOAT64_MIN or np.isneginf(val):
                            clamped_val = self.FLOAT64_MIN
                        elif np.isnan(val):
                            # Handle NaN values - use a large but finite value
                            clamped_val = self.FLOAT64_MAX
                            self.get_logger().warn("NaN value detected in covariance matrix")
                        else:
                            clamped_val = float(val)
                        
                        flat_cov.append(clamped_val)
                
                # Log warning for excessive covariance (rate-limited to once per second)
                if has_excessive_values:
                    self.excessive_covariance_count += 1
                    current_time = time.time()
                    if current_time - self.last_covariance_warning_time > 1.0:
                        self.get_logger().error(
                            f"Excessive covariance detected! Max value: {max_cov_value:.2e}, "
                            f"threshold: {self.COVARIANCE_WARNING_THRESHOLD:.2e}. "
                            f"Total occurrences: {self.excessive_covariance_count}"
                        )
                        self.last_covariance_warning_time = current_time
                
                msg.covariance = flat_cov
                # Transducers (assumes fixed size of 4)
                for i, t in enumerate(packet["transducers"]):
                    msg.transducer_velocity[i] = float(t["velocity"])
                    msg.transducer_distance[i] = float(t["distance"])
                    msg.transducer_rssi[i] = float(t["rssi"])
                    msg.transducer_nsd[i] = float(t["nsd"])
                    msg.transducer_beam_valid[i] = bool(t["beam_valid"])

                self.packet_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to handle packet: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = WaterlinkedDriver()

    async def runner():
        if not await node.connect():
            node.get_logger().error("Failed to establish connection. Exiting.")
            return
        
        await node.upload_config()

        asyncio.create_task(node.listen_for_packets())

        while rclpy.ok():
            await asyncio.sleep(0.1)

        await node.disconnect()

    try:
        asyncio.run(runner())
    finally:
        rclpy.shutdown()

