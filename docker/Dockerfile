FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=glfw

# System packages:
# - build-essential/cmake/git: build pinocchio ("pin") and other C-extension deps
# - libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf: MuJoCo / Diffusion Policy rendering (doc/install.md)
# - libx11-6 libxext6 libxrender1 libxrandr2 x11-apps: X11 client libs for the MuJoCo viewer GUI
# - libhidapi-dev libusb-1.0-0-dev: pyspacemouse HID device access
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential cmake git \
    libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
    libx11-6 libxext6 libxrender1 libxrandr2 x11-apps \
    libhidapi-dev libusb-1.0-0-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python3 -m pip install --upgrade pip

WORKDIR /workspace/RoboManipBaselines

# Full source (incl. third_party/act via .dockerignore allowlist) is needed
# at build time since `pip install -e .` is an editable install.
COPY . .

# Common install + ACT policy extras (doc/quick_start.md / doc/install.md).
# --use-deprecated=legacy-resolver mirrors the documented workaround for
# long-running/conflicting resolutions.
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -e .[act] \
    && pip install --no-cache-dir -e third_party/act/detr

WORKDIR /workspace/RoboManipBaselines/robo_manip_baselines

CMD ["/bin/bash"]
