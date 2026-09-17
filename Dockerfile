# syntax=docker/dockerfile:1
#
# ROS2 Humble + Nav2 + CUDA dev image for pacer-tupac-code-ros.
# Built on an NVIDIA CUDA base so pacer_core's torch dependency (Social-STGCNN
# predictor) can use the GPU; requires the NVIDIA Container Toolkit on the host
# (see docker-compose.yml). Ubuntu 22.04 (jammy) base, not 24.04: Humble only
# ships binary packages for jammy — mocap4ros2_qualisys requires Humble.
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ARG ROS_DISTRO=humble
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ENV ROS_DISTRO=${ROS_DISTRO} \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# --- ROS2 apt repo + base tooling ---------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        ca-certificates \
        build-essential \
        git \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

# --- ROS2 base + Nav2 + Python toolchain ---------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-base \
        ros-${ROS_DISTRO}-navigation2 \
        ros-${ROS_DISTRO}-nav2-bringup \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-pip \
        python3-numpy \
        python3-scipy \
    && rosdep init \
    && rm -rf /var/lib/apt/lists/*

# torch (CUDA 12.6 wheels, matches the base image) — installed explicitly so
# rosdep doesn't pull in a mismatched build; see --skip-keys below.
# (no --break-system-packages: that PEP 668 guard doesn't exist before
# Ubuntu 24.04's pip, and jammy's system pip predates the flag entirely.)
RUN pip3 install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cu126

WORKDIR /home/${USERNAME}
COPY pacer ./src/pacer
COPY mocap4ros2_qualisys ./src/mocap4ros2_qualisys

# mocap_msgs/qualisys_driver come from the vendored src/mocap4ros2_qualisys
# (mocap4ros2/qualisys_ros2 ecosystem, not in the default ROS2 apt repos).
# qualisys_cpp_sdk is a plain CMake lib with no package.xml, so rosdep can't
# see it's satisfied locally — skipped, colcon resolves it via the workspace.
# python3-torch is skipped because it's installed above with a pinned CUDA
# build. rqt_mocap_control (Qt GUI, unused by this project) is excluded from
# the build entirely via COLCON_IGNORE.
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y \
        --skip-keys python3-torch --skip-keys qualisys_cpp_sdk

# Pin setuptools below the version that made _core_metadata's
# canonicalize_version() call pass strip_trailing_zero=... (a packaging
# >=24.1 kwarg). rosdep above pulls a newer setuptools via pip for some
# workspace python deps, which then breaks the ament_python symlink-install
# below (legacy setup.py develop/easy_install path) against the older
# apt-provided packaging — pinned last so nothing upgrades it back.
RUN pip3 install --no-cache-dir "setuptools==58.2.0"

RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && colcon build --symlink-install

# Non-root user matching the host UID/GID (punk-opc is 1000:1000), so
# /home/ros/src bind-mounted from the host in docker-compose.yml, and
# volumes like ~/.Xauthority, resolve to files the container can actually
# read/write instead of everything landing under root.
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ${USERNAME}

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
