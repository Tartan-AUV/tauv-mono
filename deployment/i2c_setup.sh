#!/bin/bash
# $1 should be the name of the user to add
usermod -aG i2c tauv
groupadd -f -r gpio
sudo usermod -a -G gpio tauv
cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/
udevadm control --reload-rules && sudo udevadm trigger
