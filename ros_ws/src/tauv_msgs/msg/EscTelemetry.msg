# EscTelemetry.msg

std_msgs/Header header

uint8 id          # ESC identifier
int32 rpm         # Rotations per minute
float32 voltage   # Voltage in volts
float32 current   # Current in amps
float32 temperature # Temperature in Celsius
uint8 fault_code  # ESC fault code, if any
