# Docker setup (unofficial)

Not provided upstream — this `Dockerfile` / `docker-compose.yml` build the
common install plus every pip-installable simulation policy extra (SARNN,
ACT, MT-ACT, Diffusion Policy, 3D Diffusion Policy, Flow Policy, ManiFlow
Policy), per [../doc/quick_start.md](../doc/quick_start.md) and
[../doc/install.md](../doc/install.md).

### Policies not included

- **GR00T, pi0** — each needs 3-4 separate virtualenvs, cloning an external
  repo (Isaac-GR00T / lerobot) outside this repository, and (GR00T) a
  flash-attn build against a live GPU. That's a different container model
  entirely, not an extension of this one.
- **TACTO, real-ur5e, real-xarm7, GELLO** — tactile-sensor / real-robot /
  teleop-device environments that assume physical hardware on the host.
- **Isaac Gym environments** — require Python 3.6-3.8, incompatible with this
  image's Python 3.10.

Build context is the repo root (`..`), since the build needs `pyproject.toml`
and each installed policy's `third_party/<name>` sources (`act`, `roboagent`,
`eipl`, `diffusion_policy`, `3D-Diffusion-Policy`, `FlowPolicy`,
`ManiFlow_Policy`) — always run compose commands from this `docker/`
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

All pip-installable simulation policies are already built in. To add one of
the excluded ones (see "Policies not included" above):

- Extend the `RUN pip install -e .[...]` line in the `Dockerfile` with any
  new extras, add the corresponding `third_party/<name>` editable install
  and any apt packages the policy needs (see `../doc/install.md`).
- Remove the corresponding `third_party/<name>` entry from `../.dockerignore`
  so its source is included in the build context.
- Before installing a new third_party package editable, check whether its
  `setup.py` `name=` collides with one already installed in the image (see
  the `detr` / `pytorch3d` notes below) — combining previously-separate venvs
  into one image is exactly what surfaces these collisions.

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
  [doc/install.md](../doc/install.md).
- **ACT and MT-ACT both vendor a package literally named `detr`**
  (`third_party/act/detr` and `third_party/roboagent/detr` — same name,
  different DETR fork). Only `act/detr` is pip-installed; MT-ACT's
  `TrainMtAct.py`/`RolloutMtAct.py` already `sys.path.append` their own
  `third_party/roboagent` before `import detr`, which resolves correctly as
  long as `roboagent/detr` is never separately pip-installed into the same
  environment. Don't "fix" this by adding that install back — it would make
  whichever one is installed last silently shadow the other for both
  policies.
- **Diffusion Policy 3D, Flow Policy and ManiFlow Policy all vendor a
  package named `pytorch3d`.** All three only call
  `pytorch3d.ops.sample_farthest_points`, so the Dockerfile installs the
  lightweight `pytorch3d_simplified` fork (from `3D-Diffusion-Policy`) once
  and all three share it; ManiFlow's full upstream `pytorch3d` copy is
  excluded via `.dockerignore` rather than installed a second time under the
  same name.
- `pytorch3d_simplified`'s CUDA extension is built with `FORCE_CUDA=1` and a
  fixed `TORCH_CUDA_ARCH_LIST`, since no GPU is visible during
  `docker compose build` (only at `docker compose run`, via the `nvidia`
  runtime) and the build would otherwise silently fall back to a CPU-only
  extension. If the build fails on your CUDA/driver combination, either
  adjust `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability or drop both
  vars (optionally add `PYTORCH3D_FORCE_NO_CUDA=1`) for a CPU-only build.
- Diffusion Policy pulls in `robomimic==0.2.0`, which depends on an old
  `gym` release with a broken `requires.txt` (`opencv-python>=3.` isn't a
  valid version specifier) — if `pip install` fails on that, see the fix in
  [doc/install.md](../doc/install.md#diffusion-policy).
