#!/usr/bin/env python3

import sys
import time
sys.path.insert(0, 'src')

from src import UnifiedI2CDriver, UNITS_mbar, UNITS_Centigrade

def main():
    # Initialize the unified driver
    # Default: bus 1, PCA9685 at 0x40, MS5837 auto-detect model
    driver = UnifiedI2CDriver(bus_number=1, pca9685_address=0x40)
    
    # Initialize both devices
    print("Initializing devices...")
    ms5837_ok = driver.init_ms5837()
    pca9685_ok = driver.init_pca9685()
    
    if not ms5837_ok:
        print("Warning: MS5837 initialization failed")
    else:
        print("MS5837 initialized successfully")
    
    if not pca9685_ok:
        print("Warning: PCA9685 initialization failed")
    else:
        print("PCA9685 initialized successfully")
    
    if not (ms5837_ok or pca9685_ok):
        print("Error: Both devices failed to initialize")
        return
    
    # Set PWM frequency for servo control (50Hz is standard for servos)
    if pca9685_ok:
        driver.set_pwm_frequency(50)
        print("PWM frequency set to 50Hz")
    
    # Servo position values (for 50Hz):
    # ~205 = 1ms pulse (0 degrees)
    # ~307 = 1.5ms pulse (90 degrees)
    # ~409 = 2ms pulse (180 degrees)
    servo_min = 205
    servo_mid = 307
    servo_max = 409
    
    print("\nStarting test loop (Ctrl+C to exit)...")
    servo_channel = 0
    
    try:
        while True:
            # Read pressure sensor
            if ms5837_ok:
                if driver.read_depth_sensor():
                    pressure = driver.get_pressure(UNITS_mbar)
                    temperature = driver.get_temperature(UNITS_Centigrade)
                    depth = driver.get_depth()
                    
                    print(f"\nPressure: {pressure:.2f} mbar")
                    print(f"Temperature: {temperature:.2f} °C")
                    print(f"Depth: {depth:.2f} m")
                else:
                    print("Failed to read depth sensor")
            
            # Control servo based on a simple pattern
            if pca9685_ok:
                # Move servo to center
                print(f"Moving servo on channel {servo_channel} to center position")
                driver.set_pwm(servo_channel, servo_mid)
                time.sleep(1)
                
                # Move servo to min position
                print(f"Moving servo to min position")
                driver.set_pwm(servo_channel, servo_min)
                time.sleep(1)
                
                # Move servo to max position
                print(f"Moving servo to max position")
                driver.set_pwm(servo_channel, servo_max)
                time.sleep(1)
                
                # Back to center
                driver.set_pwm(servo_channel, servo_mid)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        
        # Stop the servo (set PWM to 0)
        if pca9685_ok:
            print("Stopping servo...")
            driver.set_pwm(servo_channel, 0)
        
        # Close the I2C bus
        driver.close()
        print("I2C bus closed")

if __name__ == "__main__":
    main()
