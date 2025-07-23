# General

TartanAUV (Tartan Autonomous Underwater Vehicle Team, TAUV) is an autonomous underwater robotics team at Carnegie Mellon University.

TartanAUV participates in the RoboSub competiton, which is an underwater autonomy challenge conducted in an outdoor pool.

TartanAUV's current flagship vehicle (robot) is called Osprey. In the past, we used other vehicles, namely Kingfisher and Albatross.

This is a monorepo holding all of TartanAUV's vehicle code and internal tooling. The repo is structured as follows:
- containers: vehicle, development, and CI docker containers and associated files
- firmware: firmware for vehicle microcontrollers and HDL sources for FPGAs.
- ros_ws/src: ROS2 jazzy workspace for vehicle code. May contain some nodes that are never run on the vehicle, eg. for teleop and simulation.
- deployment: tools for deploying code onto the vehicle.
- tools: assorted tools that are NOT run on the vehicle. Includes tools for vision training, CAD imports etc.

When working with this codebase, it is essential to write clear, concise, and expressive code.

You should focus on adding three types of comments when you write code:
- API docs: Module/File, Class, Function docstrings describing inputs and outputs.
- Flow overview: Header comments over individual sections to make the code easier to navigate
- Intent comments: If the purpose of some code is not clear from the immediate context to a knowledgeable reader,
  explain why this code is there.

Do your best to avoid having to write comments explaining WHAT the code is doing. Your code should be self-documenting.

# Osprey
Osprey AUV is an underwater robot. It comprises the pressurized hull, two external pressurized camera tubes, and several external sensors and actuators.

## Electronics
The main onboard compute is an NVIDIA Jetson Orin AGX running Jetpack 6.2. All TartanAUV software runs in an Ubuntu 24.04-based docker container.
Low-level control and data-acquisition tasks are delegated to a custom real-time vehicle controller board (referred to as RTVC, or Vehicle Controller). It is an STM32F767-based board
running FreeRTOS. There is an onboard Ethernet network connecting the RTVC and the Jetson. Additionally, during testing, the onboard network
may be connected to the surface via an Ethernet tether and / or a wireless connection when the robot is surfaced.

## Sensors
The AUV relies on a suite of sensors to estimate its own state and perceive the surroundings.
- Two Lucid Vision ATX162S cameras with Fisheye lenses form a stereopair. They are streaming raw images to the Jetson via dedicated point-to-point 10GbE connections
  to a PCIe NIC.
- Waterlinked A50 Doppler Velocity Log (DVL sensor) connected to the onboard Ethernet network.
- A Movella XSens MTi-300 Attitude and Heading Reference System (9-DoF IMU, connected to RTVC)
- A BlueRobotics Bar-02 pressure sensor (also called depth sensor), mounted externally and connected to RTVC

## Powertrain
The powertrain comprises 8 BlueRobotics T-200 thrusters, with four vertical and four horizontal thrusters. Each thruster is designated with three letters, as follows:
Front/Aft, Left/Right, Vertical/Horizontal.
For example, alv - aft left vertical.

## Actuators
We have the following actuators:
- Torpedo launcher ("launcher") - a spring loaded launcher pointed forward and loaded with two unpowered torpedoes. Controlled with a single servo, can launch in any order.
- Marker dropper ("dropper") - a pair of compartments with a single servo-actuated door on the bottome containing heavy marker balls that can be dropped by opening the door. The markers can be dropped sequentially in any order.
- Suction tube ("tube") - an acrylic tube with a 9th thruster mounted on a 1-DoF robotic arm for grabbing small objects by suction. It is made of clear acrylic (we can see what's inisde with the cameras). While the arm is 1 DoF (actuated by a single servo), it has two rotary joints connected by a virtual two-bar linkage. The tube can be stowed under the hull to clear the cameras' view, or deployed in front of the sub to suck in objects. The arm is spring-loaded, so only the two extreme configurations are available.

# Conventions
## Naming
### Frames of reference
Frames of reference used across multiple files should be named consistently. Frames used within one file must not shadow global frames names.

Each frame must have a descriptive name, single word whenever possible, in snake_case. Additionally, a single uppercase letter may be assigned a frame. If a letter is used on a global frame, it must be consistent. When letters are used instead of full names, document their meaning.

NED frames must be used throughout. All frames are NED unless specified otherwise.

### Points in space
A point is denoted as a lowercase letter or one or multiple lowercase words in snake_case. Point names and abbreviations are always local to a file. Note that
if a letter is used to denote a reference frame, the same lowercase letter is reserved for the origin of that reference frame.

### Transforms between frames
A transform from frame A to frame B strictly means an isometry transforming vectors from frame B to frame A, and must be denoted as `A_T_B`. If full frame names are used, we write `vehicle_T_tube_intake`.

### Rotations
A rotation from A to B rotates vectors from frame B into frame A. We write rotations as `A_R_B` if the underlying type is a matrix or a generic SO(3) object, and `A_q_B` if the underlying type is a quaternion. Euler angles must not be used under any circumstances. Angle-axis representations can be used to construct or interpret rotations.

### Positions and translations
A general translation from point `a` to point `b` expressed in frame `C` is denoted `r_ab_C`, or `r_starting_point__end_point_C`. Note the double underscore.
A position of point `a` in frame `B` is written as: `r_a_B`. If you need to use these notations, assign a leter to the frame.

### Twists
We refer to spatial velocities as twist, and express them as 6-vectors with the first three components representing linear velocity, and the last three representing the angular velocity in the body frames. We denote the twist of frame `object` as seen in frame `A` as `V_object_A`.

### Velocities
To describe linear velocity of point `a` expressed in frame `B`, we write `v_a_B`. To describe the angular velocity of frame B relative to frame A expressed in frame C, write `omega_AB_C`. This is the rate of change of `A_R_B` expressed in `C`.

### Rates of change
Rate of change of translation from `a` to `b` expressed in `C` is denoted as `r_ab_dC_C`. An acceleration of point `b` relative to `a` as seen in frame `C` expressed in frame `D` is written as `r_ab_dCdC_D`. Angular accelerations are written as `omega_AB_dC_C`.

### Forces
A force and a torque acting on point `p` expressed in frame `A` are denoted `f_p_A` and `tau_p_A`. A force-torque pair is always referred to as a "wrench", and is denoted `V_p_A`. Force source can be added, eg. `f_buoyancy_p_A`.

## Representations
In python, you should use `spatialmath-python` library wherever possible. Rotations, transforms, and twists should be represented as `SO3`, `SE3`, and `Twist3` instances. You must never use raw numpy matrices to represent rotations or transforms. Use numpy arrays for points, translations, and accelerations.

# spatialmath-python quick reference

Core Classes

- SE3: 3D poses (4x4 homogeneous matrices), represents position + orientation
- SO3: 3D rotations (3x3 orthogonal matrices)
- UnitQuaternion: 3D rotations as unit quaternions
- SE2/SO2: 2D equivalents

Import and Basic Usage

from spatialmath import SE3, SO3, UnitQuaternion
import spatialmath.base as smb  # Low-level functions

## Construction

### Identity
T = SE3()           # 4x4 identity
R = SO3()           # 3x3 identity

### From angles
R = SO3.Rx(0.5)     # Rotation about X-axis, 0.5 rad
T = SE3.Rx(0.5)     # Pure rotation as SE3
T = SE3(1,2,3)      # Pure translation
T = SE3(1,2,3) * SE3.Rx(0.5)  # Translation then rotation

### From numpy arrays
T = SE3(H)          # From 4x4 numpy array H
R = SO3(R_matrix)   # From 3x3 numpy array

## Array/Sequence Support

### Multiple poses - acts like a list
T = SE3([T1, T2, T3])  # Sequence of 3 poses
T.append(T4)           # Add pose
len(T)                 # Returns 4
T[0]                   # First pose

## Numpy Conversion

### To numpy
H = T.A                # 4x4 numpy array (property)
R_matrix = R.A         # 3x3 numpy array
t = T.t                # 3D translation vector
R_from_T = T.R         # 3x3 rotation part as numpy

### From numpy using constructors above

## Transform Composition & Application

### Composition (matrix multiplication)
T3 = T1 * T2          # SE3 * SE3 -> SE3
R3 = R1 * R2          # SO3 * SO3 -> SO3

### Point transformation
p_new = T * p         # Transform 3D point (numpy array)
points = T * points   # Transform multiple points (3xN or Nx3)

### Inverse
T_inv = T.inv()       # Same as T**-1

## Vectorized Operations

### If T has N poses and R is single rotation:
result = T * R        # Each T[i] multiplied by R
result = R * T        # R multiplied by each T[i]

### Element-wise if both have same length:
result = T1 * T2      # T1[i] * T2[i] for each i

## Rotations & Conversions

### Euler angles (roll-pitch-yaw, ZYX order)
rpy = R.rpy()         # To roll-pitch-yaw
R = SO3.RPY(r,p,y)    # From roll-pitch-yaw

### Quaternions
q = R.UnitQuaternion()  # SO3 -> UnitQuaternion
R = q.SO3()            # UnitQuaternion -> SO3
q_coeffs = q.A         # [w,x,y,z] numpy array

### Axis-angle
R = SO3.AngleAxis(angle, axis)  # From angle-axis
(angle, axis) = R.angvec()      # To angle-axis

## Adjoints & Jacobians

### Adjoint matrix (6x6 for SE3)
Ad = T.Ad()           # SE3 adjoint matrix

### Jacobians for exponential coordinates
J = SE3.jacobian(xi)  # Right Jacobian for SE3.Exp(xi)

## Low-Level Functions (spatialmath.base)

### Direct matrix operations (faster, less safe)
H = smb.transl(1,2,3) @ smb.rotx(0.5)  # 4x4 homogeneous matrix
R = smb.rotx(0.5)     # 3x3 rotation matrix

## Key Properties

- All classes validate matrix membership in their group (orthogonality, etc.)
- Use * for composition, not @ (matrix multiplication)
- Classes are list-like for sequences of transforms
- .A property gives underlying numpy array
- Supports symbolic computation with SymPy

## Common Patterns

### Chain of transformations
T_final = SE3(x,y,z) * SE3.RPY(r,p,y) * SE3.Rx(offset)

### Transform points
points_transformed = T * points_world

### Relative transformation
T_rel = T1.inv() * T2  # Transform from frame 1 to frame 2
