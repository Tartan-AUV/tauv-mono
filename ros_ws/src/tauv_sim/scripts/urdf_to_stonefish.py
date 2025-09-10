#!/usr/bin/env python3
"""
URDF to Stonefish Scenario Converter

Converts a URDF robot description to Stonefish simulator XML scenario format.
Supports configuration via YAML files for sensor and actuator parameters.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import dacite
import numpy as np
import yaml
from spatialmath import SE3, SO3


@dataclass
class Transform:
    """Represents a 3D transformation (position + orientation)"""

    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]

    @classmethod
    def from_se3(cls, T: SE3):
        """Convert from spatialmath SE3"""
        return cls(xyz=T.t.tolist(), rpy=T.rpy(unit='rad', order='zyx').tolist())

    @classmethod
    def from_urdf_elem(cls, origin_elem):
        """Parse transform from URDF element with xyz and rpy attributes"""
        xyz_str = origin_elem.get('xyz')
        rpy_str = origin_elem.get('rpy')

        xyz_parts = xyz_str.split()
        rpy_parts = rpy_str.split()

        # Ensure exactly 3 elements
        if len(xyz_parts) != 3:
            raise ValueError(f"Invalid xyz string: {xyz_str}")
        if len(rpy_parts) != 3:
            raise ValueError(f"Invalid rpy string: {rpy_str}")

        xyz = (float(xyz_parts[0]), float(xyz_parts[1]), float(xyz_parts[2]))
        rpy = (float(rpy_parts[0]), float(rpy_parts[1]), float(rpy_parts[2]))

        return cls(xyz=xyz, rpy=rpy)

    def to_stonefish_attrs(self) -> dict[str, str]:
        """Convert to Stonefish XML attributes"""
        xyz_str = f"{self.xyz[0]} {self.xyz[1]} {self.xyz[2]}"
        rpy_str = f"{self.rpy[0]} {self.rpy[1]} {self.rpy[2]}"
        return {"xyz": xyz_str, "rpy": rpy_str}

    def to_se3(self) -> SE3:
        """Convert to spatialmath SE3"""
        return SE3.Rt(SO3.RPY(self.rpy, unit='rad', order='zyx'), self.xyz)


@dataclass
class InertialData:
    """Represents inertial properties (mass, center of gravity, inertia)"""

    mass: float
    part_T_cg: Transform
    inertia: np.ndarray

    @classmethod
    def from_urdf_inertial(cls, inertial_elem):
        """Parse inertial data from URDF inertial element"""

        # Extract mass
        mass_elem = inertial_elem.find('mass')
        mass = float(mass_elem.get('value'))

        # Extract center of gravity
        origin_elem = inertial_elem.find('origin')

        xyz_str = origin_elem.get('xyz')
        rpy_str = origin_elem.get('rpy')

        xyz_parts = xyz_str.split()
        rpy_parts = rpy_str.split()

        part_T_cg = Transform.from_urdf_elem(origin_elem)

        # Extract inertia (just diagonal elements for Stonefish)
        inertia_elem = inertial_elem.find('inertia')

        ixx = float(inertia_elem.get('ixx'))
        ixy = float(inertia_elem.get('ixy'))
        ixz = float(inertia_elem.get('ixz'))
        iyy = float(inertia_elem.get('iyy'))
        iyz = float(inertia_elem.get('iyz'))
        izz = float(inertia_elem.get('izz'))

        inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

        return cls(mass=mass, part_T_cg=part_T_cg, inertia=inertia)

    def is_valid(self) -> bool:
        """Check if inertial data has meaningful values (mass > 0)"""
        return self.mass > 0.0


@dataclass
class MeshPart:
    """Represents a mesh part with its transformation and material"""

    name: str
    filename: str
    transform: Transform
    material_name: str = "Aluminium"
    look_name: str = "Red"


@dataclass
class SensorConfig:
    """Configuration for a sensor"""

    type: str
    rate: float
    parameters: dict


@dataclass
class ThrusterConfig:
    """Configuration for deadband thruster"""

    max_setpoint: float
    inverted_setpoint: bool
    normalized_setpoint: bool
    propeller_diameter: float
    propeller_right: bool
    propeller_mesh: str
    thrust_coeff_forward: float
    thrust_coeff_reverse: float
    deadband_lower: float
    deadband_upper: float


@dataclass
class RobotConfig:
    """Configuration for the robot conversion"""

    name: str
    material: str
    look: str
    sensors: dict[str, SensorConfig]
    thruster: ThrusterConfig
    thruster_setpoint_topic: str
    servo_setpoint_topic: str
    thruster_state_topic: str
    base_link_mass_override: float | None
    include_arm: bool


class URDFParser:
    """Parser for URDF files - simplified for known Osprey structure"""

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

    def get_sensor_frames(self) -> dict[str, Transform]:
        """Extract sensor frame transforms from joints"""
        frames = {}

        # Find DVL frame
        for joint in self.root.findall('joint'):
            if joint.get('name') == 'os/dvl_frame':
                origin = joint.find('origin')
                frames['dvl'] = Transform.from_urdf_elem(origin)

            elif joint.get('name') == 'os/depth_frame':
                origin = joint.find('origin')
                frames['pressure'] = Transform.from_urdf_elem(origin)

            elif joint.get('name') == 'os/imu_frame':
                origin = joint.find('origin')
                frames['imu'] = Transform.from_urdf_elem(origin)

        return frames

    def get_thruster_frames(self) -> list[tuple[str, Transform]]:
        """Extract thruster frame information in the standard order"""
        # Define thruster order to match expected layout
        thruster_names = [
            'os/thruster/flh',
            'os/thruster/flv',
            'os/thruster/alv',
            'os/thruster/alh',
            'os/thruster/arh',
            'os/thruster/arv',
            'os/thruster/frv',
            'os/thruster/frh',
        ]

        thrusters = []
        for thruster_name in thruster_names:
            # Find the joint for this thruster
            for joint in self.root.findall('joint'):
                if joint.get('name') == f'{thruster_name}_frame':
                    origin = joint.find('origin')
                    transform = Transform.from_urdf_elem(origin)
                    # Extract short name (e.g., 'flh' from 'os/thruster/flh')
                    short_name = thruster_name.split('/')[-1]
                    thrusters.append((short_name, transform))
                    break
            else:
                raise ValueError(f"Thruster {thruster_name} not found")

        return thrusters

    def get_arm_info(self) -> dict:
        """Extract arm information from URDF"""
        arm_info = {
            'arm_base_transform': None,
            'arm_base_joint': None,
            'arm_link_joint': None,
            'arm_link_visuals': {},  # Add visual transforms for arm links
        }

        # Find the arm base visual in os_hull link
        hull_link = self.root.find(".//link[@name='os_hull']")
        if hull_link is not None:
            for visual in hull_link.findall('visual'):
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None and 'os_arm_base' in mesh.get('filename', ''):
                        origin = visual.find('origin')
                        arm_info['arm_base_transform'] = Transform.from_urdf_elem(origin)
                        break

        # Find arm link visual transforms
        arm_link = self.root.find(".//link[@name='os_arm_link']")
        if arm_link is not None:
            visual = arm_link.find('visual')
            if visual is not None:
                origin = visual.find('origin')
                arm_info['arm_link_visuals']['os_arm_link'] = Transform.from_urdf_elem(origin)

        arm_tube = self.root.find(".//link[@name='os_arm_tube']")
        if arm_tube is not None:
            visual = arm_tube.find('visual')
            if visual is not None:
                origin = visual.find('origin')
                arm_info['arm_link_visuals']['os_arm_tube'] = Transform.from_urdf_elem(origin)

        # Find arm joints
        for joint in self.root.findall('joint'):
            if joint.get('name') == 'os/arm/base':
                arm_info['arm_base_joint'] = joint
            elif joint.get('name') == 'os/arm/link':
                arm_info['arm_link_joint'] = joint

        return arm_info

    def get_hull_mesh_parts(self) -> list[MeshPart]:
        """Extract all mesh parts from the os_hull link"""
        mesh_parts = []
        part_name_counts = {}  # Track part name counts to avoid duplicates

        hull_link = self.root.find(".//link[@name='os_hull']")
        if hull_link is None:
            return mesh_parts

        # Extract all visual elements with mesh geometry
        for i, visual in enumerate(hull_link.findall('visual')):
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    filename = mesh.get('filename', '')
                    if filename:
                        # Extract mesh filename from package path
                        if 'package://' in filename:
                            filename = filename.replace('package://', '')
                        if filename.startswith('assets/'):
                            filename = filename[7:]  # Remove 'assets/' prefix

                        # Skip frame meshes
                        if filename == 'frame.stl':
                            continue

                        # Create part name from filename
                        base_part_name = filename.replace('.stl', '').replace('.obj', '')
                        if not base_part_name:
                            base_part_name = f"part_{i}"

                        # Make part name unique
                        if base_part_name in part_name_counts:
                            part_name_counts[base_part_name] += 1
                            part_name = f"{base_part_name}_{part_name_counts[base_part_name]}"
                        else:
                            part_name_counts[base_part_name] = 0
                            part_name = base_part_name

                        # Get transform
                        origin = visual.find('origin')
                        transform = Transform.from_urdf_elem(origin)

                        # Determine material and look based on mesh name
                        material_name = "Aluminium"
                        look_name = "Red"

                        # Special cases for different materials/looks
                        if 'launcher' in part_name:
                            look_name = "Orange"
                        elif 'dropper' in part_name:
                            look_name = "Gray"
                        elif 'thruster' in part_name:
                            look_name = "Gray"
                        elif 'frame' in part_name:
                            look_name = "Gray"
                        elif 'arm_base' in part_name:
                            look_name = "Yellow"
                        elif 'hull' in part_name:
                            look_name = "Red"

                        mesh_part = MeshPart(
                            name=part_name,
                            filename=filename,
                            transform=transform,
                            material_name=material_name,
                            look_name=look_name,
                        )
                        mesh_parts.append(mesh_part)

        return mesh_parts

    def get_link_inertial_data(self) -> dict[str, InertialData]:
        """Extract inertial data (mass, CoG, inertia) from all links"""
        inertial_data = {}

        for link in self.root.findall('link'):
            link_name = link.get('name')
            if link_name:
                inertial_elem = link.find('inertial')
                inertial = InertialData.from_urdf_inertial(inertial_elem)
                inertial_data[link_name] = inertial

        return inertial_data


class StonefishGenerator:
    """Generator for Stonefish scenario XML"""

    def __init__(self, config: RobotConfig):
        self.config = config

    def _format_mesh_path(self, filename: str) -> str:
        """Format mesh filename with the correct ROS path"""
        if not filename:
            return filename

        # If already a full path (starts with $(find or absolute path), return as-is
        if filename.startswith('$(find') or filename.startswith('/'):
            return filename

        # Extract just the filename if it contains path separators
        if '/' in filename:
            filename = filename.split('/')[-1]

        # Add the ROS package path prefix
        return f"$(find tauv_sim)/data/osprey/assets/{filename}"

    def create_scenario(self, urdf_parser: URDFParser) -> ET.Element:
        """Create robot-specific scenario for inclusion in larger simulations"""
        scenario = ET.Element('scenario')

        # Only add the robot definition - no environment, materials, or solver
        # This allows the scenario to be included in larger simulation setups
        self._add_robot(scenario, urdf_parser)

        return scenario

    def _add_robot(self, scenario: ET.Element, urdf_parser: URDFParser):
        """Add robot definition"""
        robot = ET.SubElement(scenario, 'robot')
        robot.set('name', self.config.name)
        robot.set('fixed', 'false')
        robot.set('self_collisions', 'false')

        # Add base link
        self._add_base_link(robot, urdf_parser)

        # Add sensors
        self._add_sensors(robot, urdf_parser)

        # Add thrusters
        self._add_thrusters(robot, urdf_parser)

        # Add arm if present
        if self.config.include_arm:
            self._add_arm(robot, urdf_parser)

        # Add ROS interfaces
        self._add_ros_interfaces(robot)

        # Add world transform
        world_transform = ET.SubElement(robot, 'world_transform')
        world_transform.set('xyz', '0.0 0.0 0.0')
        world_transform.set('rpy', '0.0 0.0 0.0')

    def _add_base_link(self, robot: ET.Element, urdf_parser: URDFParser):
        """Add base link definition as compound body"""
        # Get hull mesh parts from URDF
        hull_parts = urdf_parser.get_hull_mesh_parts()

        base_link = ET.SubElement(robot, 'base_link')
        base_link.set('name', 'os_hull')
        base_link.set('type', 'compound')
        base_link.set('physics', 'submerged')

        # Use mass properties from URDF for the compound body
        inertial_data = urdf_parser.get_link_inertial_data()
        hull_inertial = inertial_data['os_hull']
        hull_part = None

        # Add all hull mesh parts as external parts
        for mesh_part in hull_parts:
            external_part = ET.SubElement(base_link, 'external_part')
            external_part.set('name', mesh_part.name)
            external_part.set('type', 'model')
            external_part.set('physics', 'submerged')
            external_part.set('buoyant', 'true')

            # Compound transform (position of this part relative to compound body origin)
            # Set to zero. All parts' origins match the compound body origin (os/body)
            # We just set the mesh transform
            compound_transform = ET.SubElement(external_part, 'compound_transform')
            compound_transform.set('xyz', '0.0 0.0 0.0')
            compound_transform.set('rpy', '0.0 0.0 0.0')

            # Physical mesh
            physical = ET.SubElement(external_part, 'physical')
            mesh = ET.SubElement(physical, 'mesh')
            mesh.set('filename', self._format_mesh_path(mesh_part.filename))
            mesh.set('scale', '1.0')
            mesh_origin = ET.SubElement(physical, 'origin')
            attrs = mesh_part.transform.to_stonefish_attrs()
            mesh_origin.set('xyz', attrs['xyz'])
            mesh_origin.set('rpy', attrs['rpy'])

            # Unclear what this does or if it's needed.
            part_origin = ET.SubElement(external_part, 'origin')
            part_origin.set('xyz', '0.0 0.0 0.0')
            part_origin.set('rpy', '0.0 0.0 0.0')

            # Material and look
            material = ET.SubElement(external_part, 'material')
            material.set('name', mesh_part.material_name)

            look = ET.SubElement(external_part, 'look')
            look.set('name', mesh_part.look_name)

            if mesh_part.name == 'os_hull':
                mass = ET.SubElement(external_part, 'mass')
                if self.config.base_link_mass_override is None:
                    mass.set('value', str(hull_inertial.mass))
                else:
                    mass.set('value', str(self.config.base_link_mass_override))

                inertia = ET.SubElement(external_part, 'inertia')

                # Note that the inertial parameters are defined in the
                # part frame, not the compound frame. However, we make these frames match.
                inertia_B = hull_inertial.inertia
                inertia.set('xyz', f"{inertia_B[0, 0]} {inertia_B[1, 1]} {inertia_B[2, 2]}")

                cg = ET.SubElement(external_part, 'cg')
                attrs = hull_inertial.part_T_cg.to_stonefish_attrs()
                cg.set('xyz', attrs['xyz'])
                cg.set('rpy', attrs['rpy'])
            else:
                # TODO: See if 0 actually breaks anything.
                epsilon = 1e-12
                eps_str = str(epsilon)
                mass = ET.SubElement(external_part, 'mass')
                # It looks like there's a bug in Stonefish where zero mass is not allowed
                mass.set('value', eps_str)
                inertia = ET.SubElement(external_part, 'inertia')
                inertia.set('xyz', f"{eps_str} {eps_str} {eps_str}")
                cg = ET.SubElement(external_part, 'cg')
                cg.set('xyz', f"{eps_str} {eps_str} {eps_str}")
                cg.set('rpy', f"{eps_str} {eps_str} {eps_str}")

    def _add_sensors(self, robot: ET.Element, urdf_parser: URDFParser):
        """Add sensor definitions"""
        sensor_frames = urdf_parser.get_sensor_frames()

        # Add IMU
        if 'imu' in self.config.sensors:
            self._add_imu_sensor(robot, self.config.sensors['imu'], sensor_frames['imu'])

        # Add DVL
        if 'dvl' in sensor_frames and 'dvl' in self.config.sensors:
            self._add_dvl_sensor(robot, self.config.sensors['dvl'], sensor_frames['dvl'])

        # Add pressure sensor
        if 'pressure' in sensor_frames and 'pressure' in self.config.sensors:
            self._add_pressure_sensor(
                robot, self.config.sensors['pressure'], sensor_frames['pressure']
            )

    def _add_imu_sensor(self, robot: ET.Element, config: SensorConfig, transform: Transform):
        """Add IMU sensor"""
        sensor = ET.SubElement(robot, 'sensor')
        sensor.set('name', 'imu')
        sensor.set('type', 'imu')
        sensor.set('rate', str(config.rate))

        # Origin at base
        origin = ET.SubElement(sensor, 'origin')
        attrs = transform.to_stonefish_attrs()
        origin.set('xyz', attrs['xyz'])
        origin.set('rpy', attrs['rpy'])

        # Range
        range_elem = ET.SubElement(sensor, 'range')
        params = config.parameters
        range_elem.set('angular_velocity', params.get('angular_velocity_range', '7.85 7.85 7.85'))
        range_elem.set('linear_acceleration', params.get('linear_acceleration_range', '20.0'))

        # ROS publisher
        imu_topic = config.parameters.get('ros_topic', 'sim/imu')
        ros_pub = ET.SubElement(sensor, 'ros_publisher')
        ros_pub.set('topic', imu_topic)

        # Noise
        noise = ET.SubElement(sensor, 'noise')
        noise.set('angle', params.get('angle_noise', '0.0 0.0 0.0'))
        noise.set('angular_velocity', params.get('angular_velocity_noise', '0.00175'))
        noise.set('yaw_drift', params.get('yaw_drift', '0.0'))
        noise.set('linear_acceleration', params.get('linear_acceleration_noise', '0.00589'))

        # Link
        link = ET.SubElement(sensor, 'link')
        link.set('name', 'os_hull')

    def _add_dvl_sensor(self, robot: ET.Element, config: SensorConfig, transform: Transform):
        """Add DVL sensor"""
        sensor = ET.SubElement(robot, 'sensor')
        sensor.set('name', 'dvl')
        sensor.set('rate', str(config.rate))
        sensor.set('type', 'dvl')

        # Specs
        specs = ET.SubElement(sensor, 'specs')
        params = config.parameters
        specs.set('beam_angle', params.get('beam_angle', '22.5'))
        specs.set('beam_positive_z', params.get('beam_positive_z', 'true'))

        # Range
        range_elem = ET.SubElement(sensor, 'range')
        range_elem.set('velocity', params.get('velocity_range', '10.0 10.0 5.0'))
        range_elem.set('altitude_min', params.get('altitude_min', '0.05'))
        range_elem.set('altitude_max', params.get('altitude_max', '50.0'))

        # Water layer
        water_layer = ET.SubElement(sensor, 'water_layer')
        water_layer.set('minimum_layer_size', params.get('minimum_layer_size', '10.0'))
        water_layer.set('boundary_near', params.get('boundary_near', '10.0'))
        water_layer.set('boundary_far', params.get('boundary_far', '30.0'))

        # Noise
        noise = ET.SubElement(sensor, 'noise')
        noise.set('velocity_percent', params.get('velocity_percent_noise', '0.1'))
        noise.set('velocity', params.get('velocity_noise', '0.01'))
        noise.set('altitude', params.get('altitude_noise', '0.01'))
        noise.set('water_velocity_percent', params.get('water_velocity_percent_noise', '0.1'))
        noise.set('water_velocity', params.get('water_velocity_noise', '0.1'))

        # History
        history = ET.SubElement(sensor, 'history')
        history.set('samples', params.get('history_samples', '1'))

        # Origin
        origin = ET.SubElement(sensor, 'origin')
        attrs = transform.to_stonefish_attrs()
        origin.set('xyz', attrs['xyz'])
        origin.set('rpy', attrs['rpy'])

        # Link
        link = ET.SubElement(sensor, 'link')
        link.set('name', 'os_hull')

        # ROS publisher
        dvl_topic = config.parameters.get('ros_topic', 'sim/dvl')
        ros_pub = ET.SubElement(sensor, 'ros_publisher')
        ros_pub.set('topic', dvl_topic)

    def _add_pressure_sensor(self, robot: ET.Element, config: SensorConfig, transform: Transform):
        """Add pressure sensor"""
        sensor = ET.SubElement(robot, 'sensor')
        sensor.set('name', 'pressure')
        sensor.set('rate', str(config.rate))
        sensor.set('type', 'pressure')

        # Range
        range_elem = ET.SubElement(sensor, 'range')
        params = config.parameters
        range_elem.set('pressure', params.get('pressure_range', '10000.0'))

        # Noise
        noise = ET.SubElement(sensor, 'noise')
        noise.set('pressure', params.get('pressure_noise', '0.1'))

        # History
        history = ET.SubElement(sensor, 'history')
        history.set('samples', params.get('history_samples', '1'))

        # Origin
        origin = ET.SubElement(sensor, 'origin')
        attrs = transform.to_stonefish_attrs()
        origin.set('xyz', attrs['xyz'])
        origin.set('rpy', attrs['rpy'])

        # Link
        link = ET.SubElement(sensor, 'link')
        link.set('name', 'os_hull')

        # ROS publisher
        pressure_topic = config.parameters.get('ros_topic', 'sim/pressure')
        ros_pub = ET.SubElement(sensor, 'ros_publisher')
        ros_pub.set('topic', pressure_topic)

    def _add_thrusters(self, robot: ET.Element, urdf_parser: URDFParser):
        """Add thruster actuators"""
        thrusters = urdf_parser.get_thruster_frames()

        for thruster_name, transform in thrusters:
            actuator = ET.SubElement(robot, 'actuator')
            actuator.set('name', f"os/thruster/{thruster_name}")
            actuator.set('type', 'thruster')

            # Specs
            specs = ET.SubElement(actuator, 'specs')
            specs.set('max_setpoint', str(self.config.thruster.max_setpoint))
            specs.set('inverted_setpoint', str(self.config.thruster.inverted_setpoint).lower())
            specs.set('normalized_setpoint', str(self.config.thruster.normalized_setpoint).lower())

            # Propeller
            propeller = ET.SubElement(actuator, 'propeller')
            propeller.set('diameter', str(self.config.thruster.propeller_diameter))
            propeller.set('right', str(self.config.thruster.propeller_right).lower())

            mesh = ET.SubElement(propeller, 'mesh')
            mesh.set('filename', self._format_mesh_path(self.config.thruster.propeller_mesh))
            mesh.set('scale', '1.0')

            material = ET.SubElement(propeller, 'material')
            material.set('name', 'Polyamid')

            look = ET.SubElement(propeller, 'look')
            look.set('name', 'PaleBlue')

            # Rotor dynamics
            rotor = ET.SubElement(actuator, 'rotor_dynamics')
            rotor.set('type', 'zero_order')

            # Thrust model (deadband)
            thrust_model = ET.SubElement(actuator, 'thrust_model')
            thrust_model.set('type', 'deadband')

            thrust_coeff = ET.SubElement(thrust_model, 'thrust_coeff')
            thrust_coeff.set('forward', str(self.config.thruster.thrust_coeff_forward))
            thrust_coeff.set('reverse', str(self.config.thruster.thrust_coeff_reverse))

            deadband = ET.SubElement(thrust_model, 'deadband')
            deadband.set('lower', str(self.config.thruster.deadband_lower))
            deadband.set('upper', str(self.config.thruster.deadband_upper))

            # Origin
            origin = ET.SubElement(actuator, 'origin')
            attrs = transform.to_stonefish_attrs()
            origin.set('xyz', attrs['xyz'])
            origin.set('rpy', attrs['rpy'])

            # Link
            link = ET.SubElement(actuator, 'link')
            link.set('name', 'os_hull')

    def _add_arm(self, robot: ET.Element, urdf_parser: URDFParser):
        """Add arm manipulator"""
        arm_info = urdf_parser.get_arm_info()

        if not arm_info['arm_base_transform']:
            return

        # Create fixed joint from os_hull to arm base joint location
        # The arm base is not a separate link in Stonefish, it's part of os_hull
        # So we directly add the arm joints that connect to os_hull

        # Add arm link with visual transform
        visual_transform = arm_info['arm_link_visuals'].get('os_arm_link')
        self._add_arm_link(robot, 'ArmLink', 'os_arm_link', visual_transform, urdf_parser)

        # Add arm tube with visual transform
        visual_transform = arm_info['arm_link_visuals'].get('os_arm_tube')
        self._add_arm_link(robot, 'ArmTube', 'os_arm_tube', visual_transform, urdf_parser)

        # Add arm base joint (connects os_hull to ArmLink)
        if arm_info['arm_base_joint'] is not None:
            self._add_arm_joint(
                robot, arm_info['arm_base_joint'], 'os/arm/base', 'os_hull', 'ArmLink'
            )

        # Add arm link joint (connects ArmLink to ArmTube)
        if arm_info['arm_link_joint'] is not None:
            self._add_arm_joint(
                robot, arm_info['arm_link_joint'], 'os/arm/link', 'ArmLink', 'ArmTube'
            )

        # Add arm actuators
        self._add_arm_actuators(robot)

    def _add_arm_link(
        self,
        robot: ET.Element,
        link_name: str,
        urdf_link_name: str,
        visual_transform: Transform | None = None,
        urdf_parser: URDFParser | None = None,
    ):
        """Add an arm link to the robot"""
        link = ET.SubElement(robot, 'link')
        link.set('name', link_name)
        link.set('type', 'model')
        link.set('physics', 'submerged')

        # Origin
        origin = ET.SubElement(link, 'origin')
        origin.set('xyz', '0.0 0.0 0.0')
        origin.set('rpy', '0.0 0.0 0.0')

        # Physical mesh
        physical = ET.SubElement(link, 'physical')
        mesh = ET.SubElement(physical, 'mesh')

        # Set appropriate mesh based on link name
        if link_name == 'ArmLink':
            mesh.set('filename', self._format_mesh_path('os_arm_link.stl'))
        else:  # ArmTube
            mesh.set('filename', self._format_mesh_path('os_arm_tube.stl'))

        mesh.set('scale', '1.0')

        # Apply visual transform to mesh origin if provided
        mesh_origin = ET.SubElement(physical, 'origin')
        if visual_transform:
            attrs = visual_transform.to_stonefish_attrs()
            mesh_origin.set('xyz', attrs['xyz'])
            mesh_origin.set('rpy', attrs['rpy'])
        else:
            mesh_origin.set('rpy', '0.0 0.0 0.0')
            mesh_origin.set('xyz', '0.0 0.0 0.0')

        # Use mass properties from URDF if available
        if urdf_parser:
            inertial_data = urdf_parser.get_link_inertial_data()
            arm_inertial = inertial_data.get(urdf_link_name)

            if arm_inertial and arm_inertial.is_valid():
                mass = ET.SubElement(link, 'mass')
                mass.set('value', str(arm_inertial.mass))

                inertia = ET.SubElement(link, 'inertia')
                inertia.set(
                    'xyz',
                    f"{arm_inertial.inertia[0]} {arm_inertial.inertia[1]} {arm_inertial.inertia[2]}",
                )

                cg = ET.SubElement(link, 'cg')
                cg.set(
                    'xyz',
                    f"{arm_inertial.cg_xyz[0]} {arm_inertial.cg_xyz[1]} {arm_inertial.cg_xyz[2]}",
                )
                cg.set(
                    'rpy',
                    f"{arm_inertial.cg_rpy[0]} {arm_inertial.cg_rpy[1]} {arm_inertial.cg_rpy[2]}",
                )
            else:
                # Fallback to default values for arm links
                print(f"Warning: Using fallback mass properties for arm link '{urdf_link_name}'")
                mass = ET.SubElement(link, 'mass')
                mass.set('value', '0.5')

                inertia = ET.SubElement(link, 'inertia')
                inertia.set('xyz', '0.01 0.01 0.01')

                cg = ET.SubElement(link, 'cg')
                cg.set('xyz', '0.0 0.0 0.0')
                cg.set('rpy', '0.0 0.0 0.0')
        else:
            # Fallback when no URDF parser provided
            mass = ET.SubElement(link, 'mass')
            mass.set('value', '0.5')

            inertia = ET.SubElement(link, 'inertia')
            inertia.set('xyz', '0.01 0.01 0.01')

            cg = ET.SubElement(link, 'cg')
            cg.set('xyz', '0.0 0.0 0.0')
            cg.set('rpy', '0.0 0.0 0.0')

        # Material and look
        material = ET.SubElement(link, 'material')
        material.set('name', 'Aluminium')

        look = ET.SubElement(link, 'look')
        look.set('name', 'Gray')

    def _add_arm_joint(
        self,
        robot: ET.Element,
        urdf_joint: ET.Element,
        joint_name: str,
        parent_name: str,
        child_name: str,
    ):
        """Add an arm joint to the robot"""
        joint = ET.SubElement(robot, 'joint')
        joint.set('name', joint_name)
        joint.set('type', urdf_joint.get('type', 'revolute'))

        # Parent and child links
        parent = ET.SubElement(joint, 'parent')
        parent.set('name', parent_name)

        child = ET.SubElement(joint, 'child')
        child.set('name', child_name)

        # Origin - use the transform from URDF
        origin_elem = urdf_joint.find('origin')
        if origin_elem is not None:
            origin = ET.SubElement(joint, 'origin')
            origin.set('xyz', origin_elem.get('xyz', '0 0 0'))
            origin.set('rpy', origin_elem.get('rpy', '0 0 0'))

        # Axis
        axis_elem = urdf_joint.find('axis')
        if axis_elem is not None:
            axis = ET.SubElement(joint, 'axis')
            axis.set('xyz', axis_elem.get('xyz', '0 0 1'))

        # Limits
        limit_elem = urdf_joint.find('limit')
        if limit_elem is not None:
            limit = ET.SubElement(joint, 'limit')
            limit.set('effort', limit_elem.get('effort', '10'))
            limit.set('velocity', limit_elem.get('velocity', '10'))
            lower = limit_elem.get('lower', '-3.14159')
            upper = limit_elem.get('upper', '3.14159')
            limit.set('lower', lower)
            limit.set('upper', upper)

    def _add_arm_actuators(self, robot: ET.Element):
        """Add servo actuators for arm joints"""
        self._add_servo_actuator(robot, 'ArmBaseServo', 'os/arm/base')
        self._add_servo_actuator(robot, 'ArmLinkServo', 'os/arm/link')

    def _add_servo_actuator(self, robot: ET.Element, actuator_name: str, joint_name: str):
        """Add a servo actuator for a joint"""
        actuator = ET.SubElement(robot, 'actuator')
        actuator.set('name', actuator_name)
        actuator.set('type', 'servo')

        joint_elem = ET.SubElement(actuator, 'joint')
        joint_elem.set('name', joint_name)

        controller = ET.SubElement(actuator, 'controller')
        controller.set('position_gain', '1000.0')
        controller.set('velocity_gain', '100.0')
        controller.set('max_torque', '100.0')

    def _add_ros_interfaces(self, robot: ET.Element):
        """Add ROS interfaces for actuators"""
        # Subscriber
        ros_sub = ET.SubElement(robot, 'ros_subscriber')
        ros_sub.set('thrusters', self.config.thruster_setpoint_topic)
        ros_sub.set('servos', self.config.servo_setpoint_topic)

        # Publisher for robot state
        ros_pub = ET.SubElement(robot, 'ros_publisher')
        ros_pub.set('thrusters', self.config.thruster_state_topic)


def main():
    parser = argparse.ArgumentParser(description='Convert URDF to Stonefish scenario format')
    parser.add_argument('urdf_file', nargs='?', help='Input URDF file path')
    parser.add_argument('-o', '--output', help='Output scenario file path')
    parser.add_argument('-c', '--config', help='YAML configuration file path')

    args = parser.parse_args()

    # Check that urdf_file is provided
    if not args.urdf_file:
        parser.error("urdf_file is required")

    # Check input file
    if not os.path.exists(args.urdf_file):
        print(f"Error: URDF file '{args.urdf_file}' not found")
        sys.exit(1)

    # Load configuration
    config_path = args.config or 'urdf_converter_config.yaml'
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    print(config_dict)

    config = dacite.from_dict(
        data_class=RobotConfig, data=config_dict, config=dacite.Config(strict=False)
    )

    # Parse URDF
    try:
        urdf_parser = URDFParser(args.urdf_file)
    except Exception as e:
        print(f"Error parsing URDF file: {e}")
        sys.exit(1)

    # Generate Stonefish scenario
    generator = StonefishGenerator(config)
    scenario = generator.create_scenario(urdf_parser)

    # Format and write output
    ET.indent(scenario, space="    ")
    tree = ET.ElementTree(scenario)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = Path(args.urdf_file).stem
        output_path = f"{base_name}_stonefish.scn"

    # Add XML declaration
    with open(output_path, 'wb') as f:
        f.write(b'<?xml version="1.0"?>\n')
        tree.write(f, encoding='utf-8')

    print(f"Converted URDF to Stonefish scenario: {output_path}")

    if not args.config and not os.path.exists(config_path):
        print("Note: No configuration file found. Use --create-config to generate a default one.")


if __name__ == '__main__':
    main()
