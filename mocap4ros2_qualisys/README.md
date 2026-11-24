# mocap4ros2_qualisys (vendored in PACER)

**Version (2026-11-24): aligned with [qualisys_docker](https://github.com/giacomotambe/qualisys_docker) `4c2e2f1`.**

What is in this folder:
- **Driver and SDK:** `qualisys_driver` and `qualisys_cpp_sdk` from
  [CentroEPiaggio/mocap4ros2_qualisys](https://github.com/CentroEPiaggio/mocap4ros2_qualisys)
  `05dcf0b`, with the streaming patch `patches/qualisys_driver_streaming.patch`
  already applied. The driver now receives every frame QTM streams, instead of
  polling every 100 ms, so `/rigid_bodies` and `/markers` come out at the QTM
  capture rate (e.g. 200 Hz) instead of 10 Hz.
- **`patches/qualisys_driver_stream_rate.patch`** (PACER, 2026-11-24, already applied):
  adds the `stream_frequency` parameter. It is needed because QTM reaches the
  NUC over **WiFi**:
  - streaming every frame (e.g. 200 Hz) over TCP on WiFi backs up, and poses
    arrive late;
  - with `stream_frequency: 50`, QTM sends 50 frames/s, which is still 20× what
    PACER uses (it samples at 2.5 Hz);
  - `0` = every frame, as in qualisys_docker (wired link only).
- **`qualisys_driver/config/qualisys_driver_params.yaml`:**
  - QTM IP and port, plus QoS settings.
  - The port key is now `port`, the name the driver actually reads. The old
    `host_port` was silently ignored.
- **`scripts/`:**
  - `record_mocap.sh <experiment> [rigid_body ...]` records the pose of one
    or more rigid bodies into a bag in `/home/ros/bags` (`./bags` of the repo).
  - `rigid_body_filter.py` republishes `/rigid_bodies` as `/<name>/pose`.
- **`mocap_msgs`:** stays at the vendored Humble version, which the driver
  needs. On the `rolling` branch the package has been renamed
  `mocap4r2_interfaces` and is not compatible.

The PACER Docker setup is unchanged.
- **Same image and same volumes:** this folder is bind-mounted at
  `/home/ros/src/mocap4ros2_qualisys`, and `docker/entrypoint.sh` rebuilds the
  workspace at every start. A `docker compose up` (or a restart of the
  container) is enough to build the patched driver.
- **No new container:** the driver and the scripts run inside `pacer_tupac`.

```bash
docker compose up -d && docker compose exec pacer bash
ros2 launch qualisys_driver qualisys.launch.py                                  # driver
/home/ros/src/mocap4ros2_qualisys/scripts/record_mocap.sh test1 traquad omnicar # recording (second shell)
```

**Network: QTM over WiFi (setup of the car).**
- The NUC on the car and the QTM PC must be on the same network. No
  static-IP Ethernet profile is needed.
- `host_name` in the params file is the **QTM PC's IP on that network**.
- QTM may not answer `ping` (Windows firewall). What matters is that port
  22222 is reachable:

```bash
timeout 3 bash -c 'echo > /dev/tcp/<QTM_IP>/22222' && echo open
```

After starting the driver, check rate and delay (stamps are the NUC
receive time):

```bash
ros2 topic hz /rigid_bodies          # ~ stream_frequency
ros2 topic delay /rigid_bodies       # compare with QTM's frame times if in doubt
```

If the rate drops or jumps on WiFi, lower `stream_frequency`, e.g. to 20.
PACER only uses the latest pose every 0.4 s.

ROS 2 runs entirely inside the NUC's container, so only the QTM stream
crosses the WiFi. If other PCs on the same WiFi use ROS 2, give this robot its
own `ROS_DOMAIN_ID` so the topics do not mix:

```bash
ROS_DOMAIN_ID=17 docker compose -f docker-compose.nuc.yml up -d
```

The parameters are read from the install folder (`--symlink-install`), so
edits to the YAML only need a driver restart. Network setup, QTM settings and
clock synchronisation are described in the qualisys_docker README.

---

# mocap4ros_qualisys

[![GitHub Action Status](https://github.com/MOCAP4ROS2-Project/mocap4ros2_qualisys/actions/workflows/main.yaml/badge.svg)](https://github.com/MOCAP4ROS2-Project/qualisys_driver)

[![codecov](https://codecov.io/gh/MOCAP4ROS2-Project/mocap_optitrack_driver/main/graph/badge.svg)](https://codecov.io/gh/MOCAP4ROS2-Project/mocap_optitrack_driver)

Requisites: qualisys_cpp_skd

Create workspace:
```
mkdir -p mocap_ws/src && cd mocap_ws/src
```
Download qualisys repo recursively to get https://github.com/qualisys/qualisys_cpp_sdk, or set QualisysSDK_PATH:
```
git clone --recursive https://github.com/MOCAP4ROS2-Project/mocap4ros2_qualisys.git
```
Install dependencies:
```
vcs import < mocap4ros2_qualisys/dependency_repos.repos
```
Compiling workspace:
```
cd .. && colcon build --symlink-install
```
Source workspace:
```
source install/setup.bash
```

Connect to the LAN. Go to wired settings, create a new profile, and give it a cool name. Go to the IPv4 tab and choose the manual method. Then, use 192.168.254.51 as the Address and 255.255.255.0 as the Netmask. Leave the gateway empty. Click apply, select that profile, and then check that you are connected to the LAN.

Setup your qualisys configuration:
```
mocap_ws/src/mocap4ros2_qualisys/qualisys_driver/config/qualisys_driver_params.yaml
```
Launch qualisys system:
```
ros2 launch qualisys_driver qualisys.launch.py
```
Visualize in rViz:
```
ros2 launch mocap_marker_viz mocap_marker_viz.launch.py mocap_system:=qualisys
```
