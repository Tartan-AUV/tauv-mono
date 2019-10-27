#!/usr/bin/env bash
echo Enabling ip forwarding
sudo sh -c "echo 1 >/proc/sys/net/ipv4/ip_forward"

echo Enabling multicast broadcasts
sudo sh -c "echo 0 >/proc/sys/net/ipv4/icmp_echo_ignore_broadcasts"

echo Restarting procps
sudo service procps restart

echo Adjust ROS environment variables in ~/.zshrc
line='export ROS_HOSTNAME=$HOST.local'
file=~/.bashrc
if ! grep -q -x -F -e "$line" <"$file"; then
  printf '%s\n' "$line" >>"$file"
fi

line='export ROS_MASTER_URI=http://$HOST.local:11311'
file=~/.bashrc
if ! grep -q -x -F -e "$line" <"$file"; then
  printf '%s\n' "$line" >>"$file"
fi

echo Verifying avahi name resolution: \(should not time out\)
avahi-resolve-host-name ${HOST}.local

source ~/.bashrc

echo Done!