# DroneCAN Test Tools

This repository contains a basic toolset for testing and configuring DroneCAN devices.

Before running anything, activate the virtual environment:

```bash
source .venv/bin/activate
```

---

## motorsetparams.py

**Purpose:** Configure ESC parameters.

* Used when setting up new ESCs
* Pushes parameters from `clanerparams.txt` to connected ESCs
* Useful for batch configuration

Refer to the Hargrave documentation for detailed parameter explanations and tuning guidance.

---

## motorcontrol.py

**Purpose:** CLI tool for running thrusters.

* Direct command line control of motors
* Useful for bench testing
* Good for verifying ESC response and direction

---

## DroneCAN GUI Tool

The `dronecan_gui_tool` is installed in this virtual environment.

If accessing remotely from macOS:

1. Install XQuartz
2. Connect using X forwarding

```bash
ssh -X tauv@[ip]
```

3. Activate the venv and launch:

```bash
source .venv/bin/activate
dronecan-gui-tool
```

This allows you to inspect nodes, monitor parameters, and debug the CAN network visually.