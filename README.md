# TAUV-MONO

The monorepo for all Tartan-AUV vehicle code, tools, and infrastructure. The vehicle software is a ROS 2 Humble workspace that runs in Docker, both on laptops/VMs (with the Stonefish simulator) and on the sub's Jetson Orin.

## Where to start

| I want to... | Go to |
|---|---|
| Set up a dev environment and run the simulator | [docs/sim.md](docs/sim.md) |
| Look up common Docker / colcon / ROS 2 commands | [docs/commands.md](docs/commands.md) |
| Understand the Docker images, compose files, or add a dependency | [containers/README.md](containers/README.md) |
| Work on the Jetson Orin (vehicle) | [containers/README.md](containers/README.md) (use the `orin` service) |
| Learn about a ROS package | the `README.md` in that package's folder (see below) |

New to the project? Read [docs/sim.md](docs/sim.md) first, then keep [docs/commands.md](docs/commands.md) handy.

## Repository layout

```
containers/   Dockerfile, compose files, and justfile for the desktop and Orin images
docs/         Setup guides and command references
ros_ws/       ROS 2 workspace (colcon builds from here)
  src/        ROS packages, most of them git submodules (see below)
tools/        Standalone scripts and utilities (calibration, CAN testing, vision labeling, thrusters, ...)
```

The ROS packages in `ros_ws/src` are git submodules, so clone with `--recurse-submodules` (or run `git submodule update --init --recursive`).

| Folder | Contents |
|---|---|
| `tauv_core` | Launch files, message definitions, state estimation, watchdogs |
| `tauv_autonomy` | Controller, trajectory generation, mission logic |
| `tauv_drivers` | Hardware drivers (DVL, DroneCAN, KVH, Xsens, servos, depth, ...) |
| `tauv_vision` | Vision and perception nodes |
| `tauv_sim` | Stonefish simulator, worlds, and vehicle models |

Each package has, or should have, its own `README.md` with details. See also [ros_ws/README.md](ros_ws/README.md) for workspace conventions.

## Contributing

Most changes touch a single submodule, so most work happens in that submodule's repo, not in `tauv-mono`.

**Typical change (one submodule)**
1. In the submodule (e.g. `ros_ws/src/tauv_sim`), create a feature branch. Don't create a branch in `tauv-mono`.
2. Implement the feature and open a pull request into that submodule's `develop` branch.
3. Once it's merged, update the submodule pointer in `tauv-mono` and commit it directly to `develop`:
   ```bash
   cd ros_ws/src/<submodule>
   git checkout develop && git pull
   cd ../../..
   git add ros_ws/src/<submodule>
   git commit -m "Update <submodule> pointer"
   ```

**Larger change (branch `tauv-mono`)**

Create a branch in `tauv-mono` and open a PR into its `develop` only when the work:
- edits more than one submodule, or
- is a larger feature that also changes things owned by `tauv-mono` itself, such as the `Dockerfile`, compose files, or docs.

In that case, still merge each submodule's changes into its own `develop` first, then include the pointer updates in the `tauv-mono` PR.

New dependencies go in the `Dockerfile`, not in a container you set up by hand. See [containers/README.md](containers/README.md#adding-a-dependency).
