# Onshape to Stonefish Pipeline Scripts

This directory contains scripts for converting Onshape CAD models to Stonefish simulation scenarios through an automated pipeline.

## Scripts Overview

### 1. `onshape_to_stonefish_pipeline.py`
Main pipeline script that automates the complete conversion process:
- Loads Onshape API credentials from config file and passes them securely to `onshape-to-robot`
- Exports URDF and meshes from Onshape using `onshape-to-robot`
- Downsamples meshes using pymeshlab with configurable limits
- Converts URDF to Stonefish scenario format

### 2. `urdf_to_stonefish.py`
Standalone URDF to Stonefish converter that:
- Parses URDF robot descriptions
- Converts to Stonefish XML scenario format
- Supports sensors, actuators, and complex robot structures
- Handles compound bodies with multiple mesh parts

## Prerequisites

### Required Dependencies
```bash
# Install onshape-to-robot
pip install onshape-to-robot

# Install pymeshlab for mesh processing
pip install pymeshlab

# Install PyYAML for configuration
pip install PyYAML
```

### Onshape API Setup
1. Create an Onshape developer account at https://dev-portal.onshape.com/
2. Generate API access keys
3. Create a configuration file with your credentials (see Configuration section)

## Quick Start

### 1. Create Configuration File
```bash
cd ros_ws/src/tauv_sim/scripts
python3 onshape_to_stonefish_pipeline.py --create-config
```

This creates `onshape_config.yaml` with default settings.

### 2. Edit Configuration
Edit `onshape_config.yaml` and add your Onshape API credentials:
```yaml
onshape:
  api_url: https://cad.onshape.com
  access_key: Your_Access_Key_Here  # Replace with your actual key
  secret_key: Your_Secret_Key_Here  # Replace with your actual key
```

### 3. Run the Pipeline
```bash
# Run complete pipeline (default osprey directory)
python3 onshape_to_stonefish_pipeline.py

# Or specify custom osprey directory
python3 onshape_to_stonefish_pipeline.py /path/to/your/osprey/directory
```

## Configuration

The configuration file (`onshape_config.yaml`) supports the following sections:

### Onshape API Settings
```yaml
onshape:
  api_url: https://cad.onshape.com
  access_key: Your_Access_Key_Here
  secret_key: Your_Secret_Key_Here
```

### Mesh Downsampling Settings
```yaml
mesh_downsampling:
  default_max_faces: 5000
  per_mesh_limits:
    os_hull.stl:
      max_faces: 10000
    thruster.stl:
      max_faces: 1000
    # Add more mesh-specific limits as needed
```

**Note**: `max_vertices` is automatically calculated from `max_faces` using Euler's formula for closed triangular meshes (V ≈ F/2 + 2).

### Output Settings
```yaml
output:
  scenario_file: osprey.scn  # Output filename
  overwrite_existing: true   # Whether to overwrite existing files
```

## Advanced Usage

### Skip Pipeline Steps
```bash
# Skip onshape-to-robot (use existing URDF/meshes)
python3 onshape_to_stonefish_pipeline.py --skip-onshape

# Skip mesh processing (use original mesh quality)
python3 onshape_to_stonefish_pipeline.py --skip-mesh-processing

# Skip both steps (only run URDF to Stonefish conversion)
python3 onshape_to_stonefish_pipeline.py --skip-onshape --skip-mesh-processing
```

### Custom Configuration File
```bash
python3 onshape_to_stonefish_pipeline.py -c my_custom_config.yaml
```

### Standalone URDF Conversion
```bash
# Convert URDF directly to Stonefish
python3 urdf_to_stonefish.py path/to/robot.urdf -o output.scn

# Create default URDF converter config
python3 urdf_to_stonefish.py --create-config

# Use custom config for URDF conversion
python3 urdf_to_stonefish.py path/to/robot.urdf -c custom_urdf_config.yaml -o output.scn
```

## Directory Structure

The pipeline expects the following directory structure:
```
tauv_sim/
├── data/
│   └── osprey/              # Target directory for onshape-to-robot
│       ├── assets/          # Generated mesh files (.stl, .obj)
│       ├── robot.urdf       # Generated URDF file
│       └── config.json      # Onshape configuration
├── scenarios/
│   └── osprey.scn          # Generated Stonefish scenario
└── scripts/
    ├── onshape_to_stonefish_pipeline.py
    ├── urdf_to_stonefish.py
    └── onshape_config.yaml
```

## Mesh Downsampling

The pipeline automatically downsamples STL meshes to reduce computational load while preserving visual quality:

### Features
- **Configurable limits**: Set different face count limits per mesh file (vertex count calculated automatically using Euler's formula)
- **Quality preservation**: Uses quadric edge collapse decimation
- **Normal computation**: Ensures proper lighting in simulation
- **ASCII STL output**: Human-readable format with normals
- **Graceful fallback**: Continues if pymeshlab is not available

### Per-Mesh Configuration
You can set specific face count limits for individual mesh files:
```yaml
mesh_downsampling:
  per_mesh_limits:
    os_hull.stl:
      max_faces: 10000    # Main hull can have more detail
    thruster.stl:
      max_faces: 1000     # Thrusters need less detail
```

The vertex count is automatically calculated using Euler's formula for triangular meshes: `max_vertices ≈ max_faces * 0.6 + 10`

## Troubleshooting

### Common Issues

1. **onshape-to-robot not found**
   ```bash
   pip install onshape-to-robot
   ```

2. **pymeshlab not available**
   ```bash
   pip install pymeshlab
   ```
   Note: Mesh processing will be skipped if pymeshlab is not available.

3. **Invalid Onshape credentials**
   - Verify your access and secret keys in the config file
   - Check that your Onshape developer account is active

4. **URDF parsing errors**
   - Ensure the URDF file exists and is valid
   - Check that all mesh files referenced in URDF are present

5. **Permission errors**
   - Ensure scripts are executable: `chmod +x *.py`
   - Check write permissions for output directories

### Debug Mode
Add verbose output by modifying the scripts to include debug prints, or run with Python's verbose flag:
```bash
python3 -v onshape_to_stonefish_pipeline.py
```

## Examples

### Basic Workflow
```bash
# 1. Create and edit config
python3 onshape_to_stonefish_pipeline.py --create-config
# Edit onshape_config.yaml with your API keys

# 2. Run complete pipeline
python3 onshape_to_stonefish_pipeline.py

# 3. Check output
ls ../scenarios/osprey.scn
```

### Development Workflow (Skip Onshape Export)
```bash
# For iterative development, skip the slow Onshape export
python3 onshape_to_stonefish_pipeline.py --skip-onshape
```

### Custom Mesh Processing Only
```bash
# Process meshes and convert URDF, but don't fetch from Onshape
python3 onshape_to_stonefish_pipeline.py --skip-onshape
```

## Integration with ROS

The generated Stonefish scenario files are designed to work with the broader ROS ecosystem:

- **Sensor topics**: IMU, DVL, and pressure sensor data published to ROS topics
- **Actuator control**: Thruster and servo commands via ROS subscribers
- **Transform frames**: Compatible with ROS tf2 coordinate frames

### ROS Topics Generated
- `sim/imu` - IMU sensor data
- `sim/dvl` - DVL velocity and altitude data  
- `sim/pressure` - Pressure/depth sensor data
- `sim/controller/thruster_setpoint` - Thruster command input
- `sim/controller/thruster_state` - Thruster state output
- `sim/controller/arm_setpoint` - Arm servo commands

## Contributing

When modifying the scripts:

1. **Follow Python conventions**: PEP 8 style, type hints, docstrings
2. **Test thoroughly**: Verify with different URDF files and configurations
3. **Update documentation**: Keep this README current with changes
4. **Handle errors gracefully**: Provide helpful error messages and fallbacks 