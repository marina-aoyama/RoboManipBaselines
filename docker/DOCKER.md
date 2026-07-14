# Docker setup (unofficial)

Not provided upstream — this `Dockerfile` / `docker-compose.yml` build the
common install + ACT policy extras, per [../doc/quick_start.md](../doc/quick_start.md)
and [../doc/install.md](../doc/install.md).

Build context is the repo root (`..`), since the build needs `pyproject.toml`
and `third_party/act` — always run compose commands from this `docker/`
directory, or pass `-f docker/docker-compose.yml` from the repo root.

## Host prerequisites (Linux)

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, with the `nvidia` runtime registered:
  ```console
  $ sudo nvidia-ctk runtime configure --runtime=docker
  $ sudo systemctl restart docker
  ```
  (one-time host setup; check `docker info | grep Runtimes` includes `nvidia`)
- Allow the container to reach your X server once per session:
  ```console
  $ xhost +local:docker
  ```

## Build & run

```console
$ cd docker
$ docker compose build
$ docker compose run --rm robomanip
```

Inside the container you're dropped into `robo_manip_baselines/`, matching
the quick start guide:

```console
$ python ./bin/Teleop.py MujocoUR5eCable --world_idx_list 0 5 --input_device keyboard
$ python ./bin/Train.py Act --dataset_dir ./dataset/MujocoUR5eCable_<date_suffix>
$ python ./bin/Rollout.py Act MujocoUR5eCable --checkpoint ./checkpoint/Act/<...>/policy_last.ckpt --world_idx 0
```

The whole repo is bind-mounted from the host at `/workspace/RoboManipBaselines`,
so source edits (from either side) and collected datasets/checkpoints are
shared live and survive container restarts.

## Attaching from VS Code instead of a terminal

If you'd rather build/start the container from the CLI and then work inside
it from a VS Code window (instead of staying in the terminal you launched it
from):

```console
$ cd docker
$ docker compose up -d --build   # --build only needed after Dockerfile changes
```

`up -d` reuses the same `command: /bin/bash` + `tty: true` / `stdin_open: true`
from `docker-compose.yml` — bash idles on the open pty instead of exiting, so
the container keeps running in the background.

Then, with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension installed:

- Command Palette (`Ctrl+Shift+P`) → **Dev Containers: Attach to Running
  Container...**
- Select `robo_manip_baselines` (the `container_name` in the compose file).
- Open `/workspace/RoboManipBaselines` in the new window.

`network_mode: host` and the X11 socket mount still apply, so the MuJoCo
viewer / teleop GUI keep working from the attached window's terminal.

Stop the container from the CLI when done:

```console
$ docker compose down
```

## Adding more policies later

Extend the `RUN pip install -e .[...]` line in the `Dockerfile` with
additional extras (e.g. `.[act,diffusion-policy]`), add any apt packages
the policy needs (see `../doc/install.md`), and remove the corresponding
`third_party/<name>` entry from `../.dockerignore` so its source is included
in the build context.

## Keyboard teleop gotcha

`Teleop.py` reads keys through two independent mechanisms:

- **WASD / Z / X** (movement, gripper) — read globally via `pynput`
  (`KeyboardInputDevice`); works no matter which window has focus.
- **`n`** (phase advance) — read via `cv2.waitKey()`, which only sees the
  key if the small **camera/image popup window** (not the MuJoCo viewer)
  has focus.

You'll need to click the image window and press `n` **multiple times** —
once per phase (`Initial → Standby → Sync → Teleop`), each printing its own
"Press 'n' to..." message. WASD/gripper only have a visible effect once
you're actually in `TeleopPhase`.

## Notes / tradeoffs

- `privileged: true` + full `/dev/bus/usb` passthrough is the simplest way
  to get SpaceMouse HID access working without curating udev rules; swap
  for a narrower `devices:` list if that's too broad for your environment.
- `network_mode: host` is a Linux-only convenience for X11; macOS/Windows
  users need a different X11 bridge (e.g. XQuartz + socat) and should drop
  this line.
- If the `pin` (Pinocchio) pip install fails during `docker compose build`,
  the upstream docs suggest installing it via apt instead — see
  [doc/install.md](doc/install.md).
