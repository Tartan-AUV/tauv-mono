import rclpy
from rclpy.node import Node
import asyncio
import json
import time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tauv_msgs.msg import WaterlinkedDvlFrame

class WaterlinkedDVL(Node):

    JSON_PROTOCOL_VERSION = "json_v3.1"

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

        self.packet_publisher = self.create_publisher(WaterlinkedDvlFrame, "dvl/frame", 10)
        self.create_service(Trigger, "calibrate_gyro", self.calibrate_gyro_callback)

    async def connect(self):
        address = self.get_parameter("address").get_parameter_value().string_value
        port = self.get_parameter("port").get_parameter_value().integer_value

        try:
            self.reader, self.writer = await asyncio.open_connection(address, port)
            self.get_logger().info(f"Connected to {address}:{port}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect: {e}")

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
            "speed_of_sound": self.get_parameter("speed_of_sound").get_parameter_value().double_value,
            "acoustic_enabled": True,
            "dark_mode_enabled": False,
            "mounting_rotation_offset": self.get_parameter("mounting_rotation_offset_deg").get_parameter_value().double_value,
            "range_mode": self.get_parameter("range_mode").get_parameter_value().string_value,
            "periodic_cycling_enabled": self.get_parameter("periodic_cycling_enable").get_parameter_value().bool_value
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
                self.get_logger().info(f"Received response: {response}")

                try:
                    response_data = json.loads(response)
                    if response_data.get("response_to") == "set_config" and response_data.get("success") is True:
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
                # Covariance
                flat_cov = [c for row in packet["covariance"] for c in row]
                msg.covariance = flat_cov
                # Transducers (assumes fixed size of 4)
                for t in packet["transducers"]:
                    msg.transducer_velocity.append(float(t["velocity"]))
                    msg.transducer_distance.append(float(t["distance"]))
                    msg.transducer_rssi.append(float(t["rssi"]))
                    msg.transducer_nsd.append(float(t["nsd"]))
                    msg.transducer_beam_valid.append(bool(t["beam_valid"]))

                self.packet_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to handle packet: {e}")

    def calibrate_gyro_callback(self, request, response):
        if not self.writer:
            self.get_logger().error("Not connected. Cannot send calibrate_gyro command.")
            response.success = False
            response.message = "Not connected to device."
            return response

        async def send_calibration():
            try:
                self.awaiting_ack = True
                self.writer.write(b'{"command":"calibrate_gyro"}\n')
                await self.writer.drain()

                response_timeout = self.get_parameter("response_timeout").get_parameter_value().double_value
                response_line = await asyncio.wait_for(self.reader.readline(), timeout=response_timeout)
                self.awaiting_ack = False

                message = response_line.decode('utf-8').strip()
                data = json.loads(message)
                if data.get("response_to") == "calibrate_gyro" and data.get("success") is True:
                    response.success = True
                    response.message = "Gyro calibration successful."
                else:
                    response.success = False
                    response.message = data.get("error_message", "Unknown error.")

            except asyncio.TimeoutError:
                self.awaiting_ack = False
                self.get_logger().error("Timeout waiting for calibrate_gyro response")
                response.success = False
                response.message = "Timeout"
            except Exception as e:
                self.awaiting_ack = False
                self.get_logger().error(f"Failed to calibrate gyro: {e}")
                response.success = False
                response.message = str(e)

        future = asyncio.run_coroutine_threadsafe(send_calibration(), asyncio.get_event_loop())
        future.result()
        return response


def main(args=None):
    rclpy.init(args=args)
    node = WaterlinkedDVL()

    async def runner():
        await node.connect()
        asyncio.create_task(node.listen_for_packets())
        await node.upload_config()

        while rclpy.ok():
            await asyncio.sleep(0.1)

        await node.disconnect()

    try:
        asyncio.run(runner())
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
