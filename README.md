# Vision-Based Robot Localization using Fiducial Markers and OpenCV

A computer vision and embedded robotics project for **indoor robot localization** using fiducial markers, OpenCV, calibrated pose estimation, and deployment on an **STM32MP257F-DK** running OpenSTLinux.

The project starts with a benchmark of multiple marker systems, selects a practical marker pipeline, and moves the selected AprilTag-based localization pipeline onto an embedded Linux board.

> **One-line summary:**  
> A low-cost webcam and printed AprilTags are used to estimate real-time 6-DoF pose and generate robot-level localization commands on an embedded ST Linux board.

---

## Current Project Status

This project now has two completed parts:

1. **Benchmarking stage**
   - Explored 11 fiducial marker systems / variants.
   - Benchmarked 6 usable marker pipelines.
   - Compared detection reliability, distance accuracy, jitter, reprojection error, and FPS.
   - Selected **AprilTag** as the embedded localization candidate.

2. **Embedded integration stage**
   - Ported the selected AprilTag pipeline to **STM32MP257F-DK**.
   - Verified USB webcam input on OpenSTLinux.
   - Ran OpenCV AprilTag detection and `solvePnP()` pose estimation on the board.
   - Added live pose and command overlay.
   - Added HTTP MJPEG browser streaming.
   - Added `systemd` auto-start for power-on demo operation.

---

## Why This Project?

Indoor robots need a reliable way to know their position and orientation.

GPS works outdoors, but it is weak indoors because satellite signals are blocked, weakened, or reflected by buildings and walls. Other methods such as LiDAR, AI-based localization, IMU, wheel encoders, and odometry are useful, but they can be expensive, compute-heavy, or prone to drift.

This project uses **fiducial marker-based vision localization**:

```text
Printed marker + calibrated camera + OpenCV detection + solvePnP pose estimation
```

This approach is:

- Low-cost
- Explainable
- Lightweight compared to LiDAR
- Useful for controlled indoor environments
- Suitable for robot alignment, docking, and target approach

---

## Core Pipeline

For full-pose markers such as ArUco, AprilTag, STag, and Chilitags:

```text
Camera frame
→ Marker detection
→ 2D corner extraction
→ Camera calibration + marker size
→ solvePnP()
→ rvec / tvec
→ distance, pose, yaw, jitter, FPS
```

For the embedded demo:

```text
USB webcam
→ STM32MP257F-DK
→ OpenCV AprilTag detection
→ solvePnP pose estimation
→ CamX / CamZ / Distance / Yaw / FPS
→ command generation
→ HTTP MJPEG stream
→ phone/laptop browser
```

---

## Marker Systems Explored

The project explored **11 marker systems / variants**.

### Fully Benchmarked

These were implemented as full pose-estimation pipelines:

- ArUco
- AprilTag
- STag
- Chilitags

### Extended Benchmarked

These were benchmarked with different detector/output styles:

- CCTag
- JuMarker UCO

### Attempted but Excluded

These were explored but excluded from final plots due to integration limitations, unstable detection, marker generation issues, or unsuitable Linux/Python workflow:

- TopoTag
- WhyCode / WhyCon
- RUNEtag
- ARToolKitX
- JuMarker Cordoba

---

## Marker Implementation Summary

| Marker | Status | Implementation | Benchmark Type |
|---|---|---|---|
| ArUco | Fully benchmarked | OpenCV ArUco dictionary | 4-corner `solvePnP()` 6-DoF pose |
| AprilTag | Fully benchmarked + embedded candidate | OpenCV AprilTag dictionary | 4-corner `solvePnP()` 6-DoF pose |
| STag | Fully benchmarked | `stag-python` detector | 4-corner `solvePnP()` 6-DoF pose |
| Chilitags | Fully benchmarked | External C++ detector + Python wrapper | 4-corner `solvePnP()` 6-DoF pose |
| CCTag | Extended benchmarked | External C++ detector | Distance estimate from detected ellipse diameter |
| JuMarker UCO | Extended benchmarked | External C++ detector output parsed in Python | Pose values from JuMarker C++ detector |

---

## Benchmark Methodology

Each accepted benchmark run used:

- Same webcam
- Same camera calibration
- Same approximate indoor lighting
- Same printed marker size for the square-marker comparisons
- 0-degree front-facing marker alignment
- 500 frames per test
- Test distances from **0.5 m to 2.5 m**

Measured metrics included:

- Detection rate
- Pose / distance success rate
- Mean Z-distance error
- Straight-line distance error
- Z-distance jitter
- Distance jitter
- Reprojection error where available
- FPS / processing time

CSV logs were validated before plotting to remove wrong ground-truth distance labels, invalid runs, and duplicate runs.

---

## Calibration

The main calibration file is:

```text
calibration/calibration_data.npz
```

It contains:

```text
mtx  = camera matrix
dist = distortion coefficients
```

The calibration file is required because `solvePnP()` needs camera geometry to convert detected 2D image corners into real-world pose values.

Main printed square marker size used:

```text
0.097 m
```

CCTag and JuMarker UCO used their measured printed size:

```text
0.096 m
```

> The camera capture resolution should match the calibration resolution. Changing resolution after calibration can affect distance accuracy.

---

## Embedded Demo

The embedded demo runs on:

```text
Board: STM32MP257F-DK
OS: OpenSTLinux
Camera: Logitech C270 USB webcam
Marker: AprilTag 16h5 ID 0
Runtime: Python 3 + OpenCV
Auto-start: systemd
Stream: HTTP MJPEG on port 8081
```

### Embedded Demo Flow

```text
Power bank / board power
→ STM32MP257F-DK boots OpenSTLinux
→ systemd starts vision-stream.service
→ preview_pose_stream.py starts automatically
→ USB webcam captures frames
→ AprilTag is detected
→ solvePnP estimates pose
→ pose and command values are overlaid
→ browser opens the live stream
```

Example stream URL:

```text
http://<BOARD_IP>:8081
```

Example used during testing:

```text
http://10.133.71.10:8081
```

### Embedded Demo Output

The stream overlays:

- AprilTag detection status
- Marker ID
- CamX
- CamZ
- Distance
- Yaw
- FPS
- Command output

### Command Logic

The embedded script uses simple threshold-based decisions:

```text
No marker detected
→ SEARCH

Distance <= 0.35 m
→ STOP

CamX > +0.04 m
→ TURN_LEFT

CamX < -0.04 m
→ TURN_RIGHT

Yaw < -10°
→ ROTATE_LEFT

Yaw > +10°
→ ROTATE_RIGHT

Otherwise
→ FORWARD
```

Thresholds:

```python
STOP_DISTANCE_M = 0.35
CENTER_TOLERANCE_M = 0.04
YAW_TOLERANCE_DEG = 10.0
```

These thresholds act as deadbands so small pose noise does not constantly change the command output.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── requirements_current_backup.txt
├── .gitignore
│
├── benchmark/
│   ├── full_pose/
│   │   ├── aruco_pose.py
│   │   ├── aruco_pose_metrics.py
│   │   ├── apriltag_pose.py
│   │   ├── apriltag_pose_metrics.py
│   │   ├── stag_pose.py
│   │   ├── stag_pose_metrics.py
│   │   ├── chilitags_pose_live_py.py
│   │   └── chilitags_pose_metrics.py
│   │
│   ├── extended/
│   │   ├── cctag_distance_live.py
│   │   ├── cctag_distance_metrics.py
│   │   ├── jumarker_uco_live_pose.py
│   │   └── jumarker_uco_metrics.py
│   │
│   └── plotting/
│       ├── plot_results.py
│       └── plot_apriltag_datadensity_effect.py
│
├── calibration/
│   ├── calibration_data.npz
│   ├── calibration_data_backup.npz
│   ├── calibration_chilitags.yml
│   └── calibration_chilitags_backup.yml
│
├── markers/
│   ├── aruco_marker_0.png
│   ├── apriltag_16h5_id0.png
│   ├── apriltag_36h11_id0.png
│   ├── stag_hd21_id00_cropped.png
│   └── chilitag_markers/
│
├── results/
│   ├── metrics_results/
│   ├── plots/
│   ├── apriltag_datadensities/
│   └── plots_apriltag_datadensity_effect/
│
├── tools/
│   ├── generate_apriltag.py
│   ├── generate_arucomarker.py
│   ├── detect_all_tags.py
│   ├── camera_alignment_lines.py
│   ├── camera_test_index_0.jpg
│   └── dof.py
│
├── embedded_demo/
│   └── stm32mp257f_dk/
│       ├── preview_pose_stream.py
│       ├── calibration_data.npz
│       ├── README.md
│       └── systemd/
│           └── vision-stream.service
│
├── external_libs/
│   ├── external/
│   └── stag-python/
│
├── native_builds/
│   ├── chilitags_pose_live
│   ├── chilitags_pose_live.cpp
│   ├── chilitags_py.cpp
│   └── chilitags_py.cpython-314-x86_64-linux-gnu.so
│
└── archive/
    ├── main.py
    └── apriltag_robot_localization.zip
```

---

## Main Scripts

### Full-Pose Benchmark Scripts

| File | Purpose |
|---|---|
| `benchmark/full_pose/aruco_pose.py` | Live ArUco pose estimation |
| `benchmark/full_pose/apriltag_pose.py` | Live AprilTag pose estimation |
| `benchmark/full_pose/stag_pose.py` | Live STag pose estimation |
| `benchmark/full_pose/chilitags_pose_live_py.py` | Live Chilitags pose estimation |
| `benchmark/full_pose/aruco_pose_metrics.py` | ArUco benchmark metrics |
| `benchmark/full_pose/apriltag_pose_metrics.py` | AprilTag benchmark metrics |
| `benchmark/full_pose/stag_pose_metrics.py` | STag benchmark metrics |
| `benchmark/full_pose/chilitags_pose_metrics.py` | Chilitags benchmark metrics |

### Extended Benchmark Scripts

| File | Purpose |
|---|---|
| `benchmark/extended/cctag_distance_live.py` | Live CCTag distance estimation |
| `benchmark/extended/cctag_distance_metrics.py` | CCTag distance benchmark metrics |
| `benchmark/extended/jumarker_uco_live_pose.py` | Live JuMarker UCO pose display |
| `benchmark/extended/jumarker_uco_metrics.py` | JuMarker UCO benchmark metrics |

### Plotting Scripts

| File | Purpose |
|---|---|
| `benchmark/plotting/plot_results.py` | Generates main comparison plots |
| `benchmark/plotting/plot_apriltag_datadensity_effect.py` | Generates AprilTag data-density plots |

### Embedded Demo Scripts

| File | Purpose |
|---|---|
| `embedded_demo/stm32mp257f_dk/preview_pose_stream.py` | Final embedded browser-stream demo |
| `embedded_demo/stm32mp257f_dk/systemd/vision-stream.service` | systemd auto-start service |
| `embedded_demo/stm32mp257f_dk/calibration_data.npz` | Calibration file used by embedded demo |

---

## Results

Benchmark CSV files are stored in:

```text
results/metrics_results/
```

Generated comparison plots are stored in:

```text
results/plots/
```

AprilTag data-density study outputs are stored in:

```text
results/apriltag_datadensities/
results/plots_apriltag_datadensity_effect/
```

The generated plots compare marker performance across distance for:

- Detection rate
- Pose / distance success rate
- Mean Z error
- Mean straight-line distance error
- Reprojection error where available
- Z jitter
- Distance jitter
- FPS

CCTag and JuMarker UCO do not currently report OpenCV-style reprojection error in the Python benchmark wrappers, so they may be absent from the reprojection-error plot while still being included in distance error, jitter, detection, and FPS plots.

---

## AprilTag Data-Density Study

The repository contains AprilTag 16h5 and 36h11 material to observe how marker data density affects pose estimation when physical marker size is fixed.

This study was useful because marker internal data density can affect long-range corner detection and pose accuracy.

---

## Running the Embedded Demo on STM32MP257F-DK

Copy the embedded demo files to the board:

```text
/home/root/vision_localization/
```

Expected board-side folder:

```text
/home/root/vision_localization/
├── preview_pose_stream.py
└── calibration_data.npz
```

Install the service file at:

```text
/etc/systemd/system/vision-stream.service
```

Reload and enable:

```bash
systemctl daemon-reload
systemctl enable vision-stream.service
systemctl start --no-block vision-stream.service
```

Check status:

```bash
systemctl status vision-stream.service --no-pager
```

Open the stream:

```text
http://<BOARD_IP>:8081
```

Useful service commands:

```bash
systemctl start --no-block vision-stream.service
systemctl stop vision-stream.service
systemctl restart vision-stream.service
systemctl status vision-stream.service --no-pager
journalctl -u vision-stream.service --no-pager -n 80
```

---

## Why Embedded Linux / MPU?

The embedded demo uses an MPU board instead of only an MCU because the vision pipeline needs:

- USB webcam support
- OpenCV
- Python/C++ runtime
- File loading
- Networking
- HTTP streaming
- Linux debugging tools

An MCU is better suited for low-level motor control, PWM, and real-time sensor reading.

Intended architecture:

```text
MPU / STM32MP257F-DK = vision + localization
MCU / motor board    = motor control
```

---

## Applications

Marker-based localization is useful in controlled indoor robotics environments such as:

- Warehouse robots
- Indoor delivery robots
- AGV alignment points
- Robot docking stations
- Charging station alignment
- Lab automation robots
- Factory workcells
- Robot arm / camera calibration zones

It is especially useful when a robot must approach a known target, align accurately, dock with a station, or localize inside a controlled indoor area.

---

## Limitations

This system has practical limitations:

- Marker must be visible to the camera
- Lighting should be reasonable
- Marker must be placed in the environment
- Pose accuracy depends on calibration quality
- Marker size must match the physical printed size
- Long-distance accuracy depends on marker visibility and corner detection quality

This is not a universal replacement for LiDAR or SLAM. However, for controlled indoor localization, docking, target approach, and alignment, it is a practical low-cost solution.

---

## Next Direction

The current system proves:

```text
benchmarking
→ marker selection
→ embedded AprilTag pose estimation
→ automatic board startup
→ HTTP-based live visualization
→ command-generation logic
```

The natural continuation is connecting the generated command output to a motor controller so the localization pipeline can be used directly for robot alignment and movement.
