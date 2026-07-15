# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=glfw

# System packages:
# - build-essential/cmake/git: build pinocchio ("pin") and other C-extension deps
# - libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf: MuJoCo / Diffusion Policy rendering (doc/install.md)
# - libx11-6 libxext6 libxrender1 libxrandr2 x11-apps: X11 client libs for the MuJoCo viewer GUI
# - libhidapi-dev libusb-1.0-0-dev: pyspacemouse HID device access
# - libgomp1: runtime dep of open3d (diffusion-policy-3d / flow-policy / maniflow-policy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential cmake git \
    libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
    libx11-6 libxext6 libxrender1 libxrandr2 x11-apps \
    libhidapi-dev libusb-1.0-0-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip

WORKDIR /workspace/RoboManipBaselines

# Full source (each installed policy's third_party/<name> is excluded from
# .dockerignore below) is needed at build time since `pip install -e .` is
# an editable install: setuptools
# reads robo_manip_baselines/version.py and enumerates robo_manip_baselines/*
# for `packages.find`, and every third_party editable install below needs its
# own setup.py + sources present. Since robo_manip_baselines/ is exactly what
# changes on every iteration anyway, splitting this into a smaller "metadata
# only" COPY ahead of the big pip installs wouldn't actually save a rebuild
# for the common case — it'd only help if you only ever touched unrelated
# doc/CI files, so it's not worth the added fragility. The `--mount=type=cache`
# pip cache below is what actually keeps rebuilds fast: even when this COPY's
# cache is busted, pip reuses previously-downloaded wheels instead of
# refetching them from PyPI.
COPY . .

# Common install + every pip-installable simulation policy extra
# (doc/quick_start.md / doc/install.md): sarnn, act, mt-act, diffusion-policy,
# diffusion-policy-3d, flow-policy, maniflow-policy. GR00T, pi0, tacto and the
# real-ur5e/real-xarm7/gello_software hardware extras are intentionally left
# out — they need separate multi-venv setups, external repo clones, or
# physical hardware, not a fit for this single-container image.
# --use-deprecated=legacy-resolver mirrors the documented workaround for
# long-running/conflicting resolutions.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --use-deprecated=legacy-resolver \
    -e .[sarnn,act,mt-act,diffusion-policy,diffusion-policy-3d,flow-policy,maniflow-policy]

# Each policy's third_party editable install, per doc/install.md.
#
# - act/detr: ACT's DETR fork. mt-act (RoboAgent) vendors its own DETR fork
#   under third_party/roboagent/detr — same package name ("detr"), different
#   code. Installing both editable would make the second one silently shadow
#   the first, breaking whichever policy was installed first. RolloutMtAct.py
#   / TrainMtAct.py resolve this themselves via sys.path.append(.../third_party/roboagent)
#   before `import detr`, so third_party/roboagent/detr is intentionally never
#   pip-installed here — only its source needs to be on disk.
# - The pytorch3d_simplified fork (from 3D-Diffusion-Policy) is installed
#   once as "pytorch3d" and shared by diffusion-policy-3d, flow-policy and
#   maniflow-policy: all three only call pytorch3d.ops.sample_farthest_points,
#   which this fork provides, avoiding a 3-way package-name collision and
#   ManiFlow's much heavier full-pytorch3d CUDA build.
# - FORCE_CUDA + TORCH_CUDA_ARCH_LIST let pytorch3d's CUDA extension build
#   without a GPU visible at `docker build` time (no --gpus during build).
#   Drop them (or set PYTORCH3D_FORCE_NO_CUDA=1) for a CPU-only build.
# - pytorch3d_simplified's setup.py does `import torch` directly (needed to
#   reach torch.utils.cpp_extension), so it must build with
#   --no-build-isolation: pip's normal PEP 517 isolated build env is a bare
#   venv that doesn't see the torch already installed in this image, and the
#   build fails with "ModuleNotFoundError: No module named 'torch'" without it.
#   That in turn means it can't be installed editable (-e): with no
#   pyproject.toml and no isolation, pip falls back to setuptools' legacy
#   build backend, which never implements the PEP 660 build_editable hook
#   ("missing the 'build_editable' hook" error). Nothing needs to live-edit
#   this vendored, compiled leaf package, so a plain (non-editable) install
#   is fine — it only needs the build_wheel hook, which legacy setuptools
#   does support.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e third_party/act/detr \
    && pip install -e third_party/eipl \
    && pip install -e third_party/diffusion_policy \
    && FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0" \
       pip install --no-build-isolation third_party/3D-Diffusion-Policy/third_party/pytorch3d_simplified \
    && pip install -e third_party/3D-Diffusion-Policy/3D-Diffusion-Policy \
    && pip install -e third_party/FlowPolicy/FlowPolicy \
    && pip install -e third_party/ManiFlow_Policy/ManiFlow \
    && pip install -e third_party/ManiFlow_Policy/third_party/r3m

WORKDIR /workspace/RoboManipBaselines/robo_manip_baselines

CMD ["/bin/bash"]
