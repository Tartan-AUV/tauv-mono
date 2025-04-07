# Network Configuration for Osprey

This folder contains network configuration files for Osprey.

Osprey has four local networks:
- `osprey-lan-0`: Subnet 10.0.0.XXX. Includes Jetson's own network interface (`eno1`), RTVC, bottom-facing OAK-D, and surface teleop computers + router
- `osprey-lan-1`: Subnet 10.0.1.XXX. PCIe NIC port 1, external camera 1
- `osprey-lan-2`: Subnet 10.0.2.XXX. PCIe NIC port 2, external camera 2
- WiFi card (todo)
