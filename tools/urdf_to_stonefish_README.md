# URDF to Stonefish Converter

A Python command-line tool for converting URDF robot descriptions to Stonefish simulator XML scenario format.

## Overview

This tool converts URDF files (specifically designed for AUV robots) into robot-specific Stonefish scenario files that can be included in larger simulations. It supports:

- **Hull conversion**: Main robot body with mesh-based physics
- **Sensor mapping**: IMU, DVL, and pressure sensors with configurable parameters
- **Thruster configuration**: Multiple thrusters with deadband thrust models
- **Manipulator support**: Basic arm structure detection (partial implementation)
- **YAML configuration**: Flexible parameter configuration for all components

## Features

### Supported Components

1. **Hull/Base Link**
   - Mesh-based physics simulation
   - Configurable mass, inertia, and center of gravity
   - Material and visual properties

2. **Sensors**
   - **IMU**: Inertial measurement unit with noise models
   - **DVL**: Doppler velocity log with water layer measurement
   - **Pressure**: Depth sensor with configurable range and noise

3. **Thrusters**
   - Deadband thrust model (configurable forward/reverse coefficients)
   - Configurable propeller mesh and properties
   - Zero-order rotor dynamics
   - ROS interface support

4. **Output Format**
   - Robot-only scenario file for inclusion in larger simulations
   - Compatible with Stonefish include mechanism
   - No environment or material definitions (assumed to be in main scenario)

## Usage

### Basic Usage

```bash
# Convert URDF to Stonefish scenario
python3 urdf_to_stonefish.py robot.urdf -o robot_scenario.scn -c config.yaml
```

### Create Default Configuration

```bash
# Generate a default configuration file
python3 urdf_to_stonefish.py --create-config -c my_config.yaml
```

### Command Line Options

- `urdf_file`: Input URDF file path
- `-o, --output`: Output scenario file path (default: `{urdf_name}_stonefish.scn`)
- `-c, --config`: YAML configuration file path (default: `urdf_converter_config.yaml`)
- `--create-config`: Create a default configuration file

## Configuration File

The tool uses a YAML configuration file to specify parameters for all components. Here's the structure:

```yaml
robot:
  name: osprey
  base_link_mesh: hull_foam_cameras_4k.obj
  mass: 23.0
  inertia: [2.10, 2.06, 3.78]
  cg: [0.038, 0.049, -0.036]
  material: Aluminium
  look: Red
  physics: submerged

sensors:
  imu:
    type: imu
    rate: 100.0
    parameters:
      angular_velocity_range: "7.85 7.85 7.85"
      linear_acceleration_range: "20.0"
      # ... other IMU parameters
  
  dvl:
    type: dvl
    rate: 7.0
    parameters:
      beam_angle: "22.5"
      velocity_range: "10.0 10.0 5.0"
      # ... other DVL parameters
  
  pressure:
    type: pressure
    rate: 100.0
    parameters:
      pressure_range: "10000.0"
      # ... other pressure parameters

thruster:
  max_setpoint: 362.0
  propeller_diameter: 0.2
  propeller_mesh: t200_prop_ccw.obj
  thrust_coeff_forward: 0.000371
  thrust_coeff_reverse: 0.000297
  deadband_lower: -47.0
  deadband_upper: 39.2

ros:
  subscriber_topic: sim/controller/thruster_setpoint
  publisher_topic: sim/controller/thruster_state
```

## How It Works

### URDF Parsing

The tool analyzes the URDF file to extract:

1. **Base Link**: Usually the first link (hull)
2. **Sensor Frames**: Dummy links with names containing "dvl", "depth", or "pressure"
3. **Thruster Frames**: Dummy links with names containing "thruster"
4. **Arm Structure**: Links and joints containing "arm" (partial support)

### Frame Mapping

- Sensor and thruster positions are extracted from URDF joint transforms
- Coordinate transformations are preserved in the Stonefish format
- Reference frames are mapped to the robot's base link

### Output Generation

The tool generates a minimal robot-specific scenario file including:

- Robot definition with sensors and actuators
- ROS interface configuration
- Proper XML structure for inclusion in larger simulations

The output does not include environment, materials, or solver settings, making it suitable for inclusion in larger simulation scenarios using Stonefish's include mechanism.

## Limitations

1. **Arm Manipulator**: Basic detection only, full conversion not implemented
2. **Mesh Paths**: Assumes mesh files are available in Stonefish data directory
3. **Sensor Types**: Limited to IMU, DVL, and pressure sensors
4. **Thruster Models**: Only deadband thrust model supported

## Example

```bash
# Create configuration
python3 urdf_to_stonefish.py --create-config -c osprey_config.yaml

# Edit configuration as needed
# ...

# Convert URDF
python3 urdf_to_stonefish.py robot.urdf -c osprey_config.yaml -o osprey_sim.scn
```

This will generate a robot-specific scenario file that can be included in larger simulations:

```xml
<!-- main_simulation.scn -->
<?xml version="1.0"?>
<scenario>
    <environment>
        <ned latitude="41.7777" longitude="3.0333"/>
        <ocean />
        <!-- environment setup -->
    </environment>
    
    <materials>
        <!-- material definitions -->
    </materials>
    
    <looks>
        <!-- look definitions -->
    </looks>
    
    <!-- Include the robot -->
    <include file="osprey_sim.scn"/>
    
    <!-- Add static bodies, other robots, etc. -->
</scenario>
```

## Dependencies

- Python 3.6+
- PyYAML
- xml.etree.ElementTree (built-in)

## Notes

- The tool filters out dummy links and frames that were used for CAD/design purposes
- All mesh references should be relative to the Stonefish data directory
- ROS topic names are configurable through the YAML file
- The generated robot scenario assumes materials (Aluminium, Polyamid) and looks (Red, PaleBlue) are defined in the main scenario
- Use Stonefish's include mechanism to incorporate the robot into larger simulations 