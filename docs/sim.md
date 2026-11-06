# Simulator & Dev Environment Setup

Set up the development environment and simulator on a fresh machine, for simulation and testing without the sub.

**Contents**
1. [VM Setup](#1-vm-setup) (skip if you're on a native Linux machine)
2. [Install Dependencies](#2-install-dependencies)
3. [GitHub Container Registry (optional)](#3-github-container-registry-optional)
4. [Clone, Get the Image, and Attach](#4-clone-get-the-image-and-attach)
5. [Build the ROS 2 Workspace](#5-build-the-ros-2-workspace)
6. [Running the Simulator](#6-running-the-simulator)

See also `containers/README.md` for details on the images and compose files.

---

## 1. VM Setup

### Install VMware Fusion
1. Go to VMware.com and download **VMware Fusion 13** (listed alongside Workstation under VMware products).
2. You will need to create a Broadcom account to download it.

### Create the virtual machine
Install Ubuntu 22.04 or newer.

| Setting | Value |
|---|---|
| Disk space | **30 GB** minimum, **40 GB** recommended |
| Memory | **8 GB** recommended (needed for the ROS build) |

To grow the disk, click the wrench icon in VMware Fusion and increase the size, then run this inside the VM:
```bash
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
```

### Enable OpenSSH
Make sure you can access the VM remotely.

### Install a GUI
Install XFCE (a lightweight GUI) so the simulator can display:
```bash
sudo apt update
sudo apt install xfce4
```

### Connect via SSH
1. Find the VM's IP address with `ip a`.
2. SSH into the machine using that IP for the remaining steps.

---

## 2. Install Dependencies

### Docker Engine
1. Install Docker Engine (including the Buildx and Compose plugins) following the official guide: [Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
2. Add your user to the `docker` group to run Docker without `sudo` (log out and back in afterwards):
   ```bash
   sudo usermod -aG docker $USER
   ```

### just (optional, recommended)
[`just`](https://github.com/casey/just) wraps the docker commands below.
```bash
sudo apt install just
```
If it isn't in your apt repos, use the install instructions in the just README. Every `just` command below has a plain `docker` equivalent.

### Git Configuration
1. Configure global Git settings. The container mounts your host `~/.gitconfig`, so this is also your identity inside the container:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
2. Set up your SSH key with GitHub so you can authenticate when cloning. If you run an SSH agent, it is forwarded into the container.

---

## 3. GitHub Container Registry (optional)

The TAUV images are hosted on the GitHub Container Registry (`ghcr.io/tartan-auv`). Logging in lets you **pull** prebuilt images instead of building them locally, and **push** images you've built so teammates (and the Orin) can pull them. If you only build images locally, you can skip this.

```bash
docker login ghcr.io
```
- **Username:** your GitHub username
- **Password:** a GitHub personal access token (classic), created at *Settings -> Developer settings -> Personal access tokens -> Tokens (classic)*
- **Scopes:** `read:packages` is enough to pull. Pushing also needs `write:packages`.

Pull and push from `containers/`:
```bash
just pull desktop      # pull the published image
just push              # build and push the release images (CI normally does this)
```

---

## 4. Clone, Get the Image, and Attach

### Clone the repository
```bash
git clone --recurse-submodules https://github.com/Tartan-AUV/tauv-mono.git
cd tauv-mono/containers
```
The ROS packages in `ros_ws/src` are git submodules. If you forgot `--recurse-submodules`, run `git submodule update --init --recursive`.

### Optional config
```bash
cp .env.example .env                                     # image tag, GPU, etc.
cp compose.override.example.yaml compose.override.yaml   # personal mounts (nvim, tmux, ...)
```
Both files are gitignored. Edit or delete anything you don't want.

### Get the desktop image
Pull the prebuilt image (fastest; requires the [GHCR login](#3-github-container-registry-optional)):
```bash
just pull desktop
```
or build it locally (needed if your host UID isn't 1000, or you changed the Dockerfile):
```bash
just build desktop
```
Without `just`:
```bash
docker compose --profile desktop pull desktop
# or
HOST_UID=$(id -u) HOST_GID=$(id -g) docker buildx bake desktop
```

### Start the container
```bash
just up desktop
```
Without `just`:
```bash
xhost +SI:localuser:$(id -un)
docker compose up -d desktop
```
`just up` also runs `xhost` so GUI apps in the container can open windows on your display.

### Attach to the container
SSH into the machine, then open a shell in the running container:
```bash
docker exec -it tauv-desktop-1 bash
```
If the name differs, find it with `docker ps`.

The repo is mounted at `/tauv-mono`, and you start in `/tauv-mono/ros_ws`. ROS 2 Humble is sourced automatically, as is the workspace overlay once it has been built.

---

## 5. Build the ROS 2 Workspace

Inside the container, from `/tauv-mono/ros_ws`:
```bash
colcon build
source install/setup.bash
```
The desktop container sets `COLCON_DEFAULTS_FILE=colcon_defaults.desktop.yaml`, which already enables `--symlink-install`, builds in Release mode, and skips hardware-only packages (DVL, DroneCAN, KVH, Xsens, servo, vision, etc.). You don't need to pass those flags yourself.

- **8 GB** of memory is recommended.
- If the build freezes, it has probably run out of memory. Restrict it to one package at a time:
  ```bash
  colcon build --parallel-workers 1
  ```

New terminals source the overlay automatically once `install/setup.bash` exists. In a terminal that was open before the first build, run `source install/setup.bash` manually.

---

## 6. Running the Simulator

```bash
ros2 launch tauv_sim main.launch.py
```

Stonefish opens a GUI window, so the container needs X11 access. `just up desktop` handles this each time; if you start the container by hand, run `xhost +SI:localuser:$(id -un)` on the host first (this resets on reboot).

Launch arguments (append as `name:=value`):

| Argument | Default | Description |
|---|---|---|
| `record` | `false` | Record a rosbag of the run |
| `tune` | `false` | Enable autotuning |

There are other launch files in `ros_ws/src/tauv_sim/launch/` for testing or verifying specific subsystems.

### NVIDIA GPU (optional)
If your machine has an NVIDIA GPU, the simulator may crash or lag without GPU passthrough.

1. Install the NVIDIA Container Toolkit on the host:
   ```bash
   sudo apt install nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
2. In `containers/.env`, enable the GPU override:
   ```
   COMPOSE_FILE=compose.yaml:compose.gpu.yaml
   ```
   (add `:compose.override.yaml` if you also use personal mounts, since setting `COMPOSE_FILE` stops compose from auto-loading it.)
3. Recreate the container: `just down && just up desktop`.

Note that the desktop image ships CPU-only PyTorch. If you need GPU torch, change the install line in the `desktop` stage of `containers/Dockerfile` and rebuild.
