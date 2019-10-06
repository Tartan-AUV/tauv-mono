# Thruster Managers and Allocators
This directory contains the thruster managers and thruster allocators.

## Published Topics
* `/manta/thrusters/{thruster number}/input/` The current input being sent to each thruster.
* `/manta/thrusters/{thruster number}/is_on/` Whether to enable this thruster or not.


## Subscribed Topics
* `/manta/thruster_manager/input` The force and torque to apply to the vessel.

## Vehicle Config Files
Config files for each vessels' thruster configuration are located in tauv_common/models/{vehicle_name}/
