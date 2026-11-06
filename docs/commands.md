# Useful Commands

## Docker

Run compose and `just` commands from `tauv-mono/containers`. Services are `desktop` (laptop/VM) and `orin` (vehicle). Containers are named `tauv-<service>-1`, e.g. `tauv-desktop-1`.

| Command | Description |
|---|---|
| `docker ps` | List running containers (add `-a` to see all containers) |
| `docker start [container name]` | Start a stopped container |
| `docker exec -it [container name] bash` | Start a shell session inside a running container |
| `docker compose up -d [service name]` | Create and start a service in the background (`desktop` or `orin`) |
| `docker compose down` | Stop and remove containers (add `--profile desktop --profile orin`, or use `just down`) |
| `docker compose pull [service name]` | Pull the published image for a service (needs a [GHCR login](sim.md#3-github-container-registry-optional)) |
| `docker buildx bake [target]` | Build an image locally (`desktop` or `orin`). Set `HOST_UID=$(id -u) HOST_GID=$(id -g)` first, or use `just build` |

`just` shortcuts (from `containers/`): `just build desktop`, `just pull desktop`, `just up desktop`, `just shell desktop`, `just down`.

## ROS 2

Run these inside the container, from `/tauv-mono/ros_ws`.

| Command | Description |
|---|---|
| `colcon build` | Build the ROS 2 workspace. Symlink install, Release mode and the skipped hardware-only packages come from `colcon_defaults.*.yaml`, so no extra flags are needed |
| `colcon build --packages-select [package name]` | Rebuild a single package |
| `colcon build --packages-up-to [package name]` | Build a package and its dependencies |
| `colcon build --parallel-workers 1` | Build one package at a time (use if the build runs out of memory) |
| `source install/setup.bash` | Source the workspace (new shells do this automatically once the workspace has been built) |
| `ros2 launch [package name] [launch file]` | Run a launch file |
| `ros2 topic list` | See a list of active topics |
| `ros2 topic hz [topic name]` | See the rate at which data is being published to a topic |
| `ros2 topic info [topic name] --verbose` | See information about a topic, such as publishers and subscribers |
| `ros2 topic echo [topic name]` | Echo incoming topic messages in the shell (use `--field` to see only a specific field) |
| `ros2 node list` | See a list of active nodes |
| `ros2 node info [node name]` | See information about a node, such as published or subscribed topics |

## Typical scenario after SSH'ing into the VM

1. Start the container (if it isn't running):
   ```bash
   docker start tauv-desktop-1
   ```
2. Enter the container:
   ```bash
   docker exec -it tauv-desktop-1 bash
   ```
3. Build the workspace (technically only necessary if you edited any files):
   ```bash
   colcon build
   ```
4. Source the workspace:
   ```bash
   source install/setup.bash
   ```
5. Launch the simulator:
   ```bash
   ros2 launch tauv_sim main.launch.py
   ```

## Other tips

- Use Tab to autocomplete container names, topic names, or to see other potential parameters.
- Use Ctrl+R to search your past bash commands.
