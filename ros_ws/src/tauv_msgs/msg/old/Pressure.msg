# Pressure.msg

std_msgs/Header header  # Measurement timestamp (center of the averaging interval)

float64 pressure        # Exterior pressure in Pascals
float64 temperature     # Exterior temperature in ???
float64 avg_window      # Interval over which the measurement was averaged
