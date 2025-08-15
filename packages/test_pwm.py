#!/usr/bin/env python3
import time
from adafruit_servokit import ServoKit
from adafruit_extended_bus import ExtendedI2C as I2C  # provides try_lock/unlock API

# Open /dev/i2c-2
i2c = I2C(7)

# PCA9685 at 0x40 (default). 16 channels.
kit = ServoKit(channels=16, i2c=i2c, address=0x40)

# Optional: set the PCA9685 frequency to 50 Hz for hobby servos
# (ServoKit defaults to 50/60 depending on version; set explicitly to be safe)
kit._pca.frequency = 50  # underscore is fine for a test script

# Send a constant pulse on channel 5
# Use pulse-width mapping: set the valid range, then set a fixed fraction.
for i in range(9, 10):
    kit.servo[i].set_pulse_width_range(1000, 2000)  # µs; adjust to your servo spec
    kit.servo[i].angle = None                      # disable angle mapping
    kit.servo[i].fraction = 0.7
    time.sleep(3)
    kit.servo[i].fraction = 0.3
    time.sleep(3)
    kit.servo[i].fraction = 0.5
    time.sleep(3)

