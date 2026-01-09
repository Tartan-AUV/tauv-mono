#!/bin/bash
sudo modprobe can
sudo modprobe can_raw
sudo modprobe mttcan
sudo ip link set can1 up type can bitrate 1000000 dbitrate 1000000 berr-reporting on fd on
