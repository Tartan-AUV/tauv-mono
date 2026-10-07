# TAUV Docker

Two images, built from one `Dockerfile`:

| Image | For | Base |
|---|---|---|
| `tauv-desktop` | laptop / VM development and desktopulation | `ubuntu:22.04` + ROS 2 Humble desktop |
| `tauv-orin` | the vehicle's Jetson Orin | NVIDIA JetPack r36.x + ROS 2 Humble |

Both share a `base` stage (ROS, colcon, rosdep, non-root user `tauv`). Everything else is
added per platform as needed.

## Files

```
Dockerfile                     stages: base -> desktop, base -> orin
docker-bake.hcl                build targets (desktop, orin, plus *-release for CI)
compose.yaml                   services shared by everyone (desktop, orin)
compose.gpu.yaml               optional NVIDIA override for desktop
compose.override.example.yaml  template for personal mounts
.env.example                   optional settings (tag, data dir, compose files)
justfile                       shortcuts for the commands below
```

## Quick start

Requires Docker with Buildx and [`just`](https://github.com/casey/just) (optional but easier).

```bash
cp compose.override.example.yaml compose.override.yaml   # optional: personal mounts
cp .env.example .env                                     # optional: settings

just build desktop      # or: just build orin   (or `just pull desktop` to use the published image)
just up desktop
just shell desktop
```

Without `just`:

```bash
HOST_UID=$(id -u) HOST_GID=$(id -g) docker buildx bake desktop
docker compose up -d desktop
docker compose exec desktop bash
```

On the Orin, use `orin` in place of `desktop`. Inside the container the workspace is at
`/tauv-mono/ros_ws`, and `/opt/ros/humble` plus the workspace overlay (if built) are sourced
automatically.

## Adding a dependency

1. Decide where it belongs: `base` (both platforms), `desktop`, or `orin`.
2. Add it to that stage's `apt-get install` list, keeping the list sorted. Use a comment if
   the reason isn't obvious.
3. Only if apt has no package, use a pinned pip package or a prebuilt wheel. Build from
   source as a last resort, in its own `RUN` with a comment explaining why.
4. Rebuild: `just build <target>`.

Packages vendored into `ros_ws/src` (for example a patched `robot_localization`) belong in the
workspace, not in the image.

## Notes

- **File ownership:** the container user's UID/GID is baked in at build time (default 1000).
  Published images use 1000. If your host UID differs, run `just build` locally so
  `build/`, `install/` and `log/` aren't owned by the wrong user.
- **GUI apps:** the desktop container uses X11 (through XWayland on Wayland desktops).
  `just up desktop` runs `xhost` so the container user can connect.
- **NVIDIA GPU on the desktop:** set `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` in `.env`.
- **Git identity** is read from your host `~/.gitconfig` (mounted read-only), so there is no
  per-user build step.
- **Cross-building the Orin image on x86** needs QEMU/binfmt set up. Building on the Orin
  itself needs nothing extra.
