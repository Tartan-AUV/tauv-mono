# WaterlinkedDvlFrame.msg
std_msgs/Header header

# Timestamp from the DVL device
float64 time

# Velocity components in m/s
float64 vx
float64 vy
float64 vz

# Figure of merit (lower is better)
float64 fom

# 3x3 covariance matrix, flattened row-major
float64[9] covariance

# Altitude above seabed in meters
float64 altitude

# Transducer-specific measurements (4 beams assumed)
float64[4] transducer_velocity
float64[4] transducer_distance
float64[4] transducer_rssi
float64[4] transducer_nsd
bool[4]    transducer_beam_valid

# Whether velocity measurement is valid
bool velocity_valid

# DVL status code
int32 status

# DVL timestamps (e.g. in microseconds since epoch or device boot time)
int64 time_of_validity
int64 time_of_transmission
